// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * This file is part of dmaplane.
 *
 * dmaplane is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License.
 *
 */

/*
 * weight_streamer_tui.c — Interactive TUI for weight streaming simulation
 *
 * ncurses-based interface for the accelerator weight streaming simulator.
 * Allows live parameter tuning and shows real-time transfer visualization.
 *
 * Build:
 *   gcc -Wall -O2 -o weight_streamer_tui weight_streamer_tui.c -lncurses
 *
 * Usage:
 *   sudo ./weight_streamer_tui [ib_device]
 *
 * Controls:
 *   Up/Down     Select parameter
 *   Left/Right  Adjust value
 *   Enter/r     Run simulation with current parameters
 *   c           Run continuous (auto-cycling epochs)
 *   s           Stop continuous mode
 *   q           Quit
 *
 * Prerequisites:
 *   - libncurses-dev installed
 *   - dmaplane.ko loaded
 *   - soft-RoCE configured
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <ncurses.h>

#include "dmaplane_uapi.h"

#define DEV_PATH      "/dev/dmaplane"
#define MAX_LAYERS    32
#define MAX_HISTORY   64

/* ── Tuneable parameters ── */

struct param {
	const char *name;
	const char *unit;
	int value;
	int min;
	int max;
	int step;
};

enum {
	P_LAYERS = 0,
	P_SHARD_KB,
	P_QDEPTH,
	P_ITERATIONS,
	P_COUNT
};

static struct param params[P_COUNT] = {
	[P_LAYERS]     = { "Model layers",      "",     8,   2,  32,  1 },
	[P_SHARD_KB]   = { "Shard size",        "KB",  64,   4, 512,  0 }, /* step=0 means power-of-2 */
	[P_QDEPTH]     = { "Queue depth",       "",     4,   1,  64,  0 },
	[P_ITERATIONS] = { "Iters per layer",   "",     1,   1,  50,  1 },
};

/* ── Per-epoch result ── */

struct epoch_result {
	int     epoch_num;
	int     layers;
	int     shard_kb;
	int     qdepth;
	double  fwd_seq_ms;
	double  fwd_seq_mbps;
	double  fwd_pipe_ms;
	double  fwd_pipe_mbps;
	double  speedup;
	double  bwd_ms;
	double  bwd_mbps;
	double  total_ms;
};

/* ── Globals ── */

static int dev_fd = -1;
static int selected_param = 0;
static int running = 0;
static int continuous = 0;
static int epoch_count = 0;
static struct epoch_result history[MAX_HISTORY];
static int history_count = 0;
static char status_msg[256] = "Press Enter to run, 'c' for continuous, 'q' to quit";
static char ib_dev_name[32] = "";

/* Per-layer tracking */
static __u32 layer_buf_ids[MAX_LAYERS];
static __u32 layer_mr_ids[MAX_LAYERS];
static int layers_allocated = 0;

/* ── Helpers ── */

static void adjust_param(int idx, int dir)
{
	struct param *p = &params[idx];
	if (p->step == 0) {
		/* Power-of-2 stepping */
		if (dir > 0 && p->value < p->max)
			p->value *= 2;
		else if (dir < 0 && p->value > p->min)
			p->value /= 2;
	} else {
		p->value += dir * p->step;
		if (p->value < p->min) p->value = p->min;
		if (p->value > p->max) p->value = p->max;
	}
}

static void pick_default_ib_device(void)
{
	DIR *dir;
	struct dirent *de;

	dir = opendir("/sys/class/infiniband");
	if (!dir)
		return;

	while ((de = readdir(dir)) != NULL) {
		if (de->d_name[0] == '.')
			continue;
		strncpy(ib_dev_name, de->d_name, sizeof(ib_dev_name) - 1);
		ib_dev_name[sizeof(ib_dev_name) - 1] = '\0';
		break;
	}

	closedir(dir);
}

static void print_available_ib_devices(FILE *stream)
{
	DIR *dir;
	struct dirent *de;
	int found = 0;

	dir = opendir("/sys/class/infiniband");
	if (!dir) {
		fprintf(stream, "Could not read /sys/class/infiniband\n");
		return;
	}

	fprintf(stream, "Available IB devices:");
	while ((de = readdir(dir)) != NULL) {
		if (de->d_name[0] == '.')
			continue;
		fprintf(stream, " %s", de->d_name);
		found = 1;
	}
	if (!found)
		fprintf(stream, " (none)");
	fprintf(stream, "\n");

	closedir(dir);
}

/* ── RDMA resource management ── */

static int setup_rdma(void)
{
	struct dmaplane_rdma_setup setup = {
		.port        = 1,
		.cq_depth    = 1024,
		.max_send_wr = 256,
		.max_recv_wr = 256,
	};
	snprintf(setup.ib_dev_name, sizeof(setup.ib_dev_name), "%s", ib_dev_name);
	return ioctl(dev_fd, IOCTL_SETUP_RDMA, &setup);
}

static void teardown_rdma(void)
{
	ioctl(dev_fd, IOCTL_TEARDOWN_RDMA, NULL);
}

static int allocate_layers(int n_layers, __u32 shard_bytes)
{
	int i;

	for (i = 0; i < n_layers; i++) {
		struct dmaplane_buf_params buf = {
			.alloc_type = BUF_TYPE_PAGES,
			.size       = shard_bytes,
			.numa_node  = DMAPLANE_NUMA_ANY,
		};
		if (ioctl(dev_fd, IOCTL_CREATE_BUFFER, &buf) < 0)
			goto fail;
		layer_buf_ids[i] = buf.buf_id;

		struct dmaplane_mr_params mr = {
			.buf_id       = buf.buf_id,
			.access_flags = DMAPLANE_IB_ACCESS_LOCAL_WRITE |
					DMAPLANE_IB_ACCESS_REMOTE_WRITE |
					DMAPLANE_IB_ACCESS_REMOTE_READ,
		};
		if (ioctl(dev_fd, IOCTL_REGISTER_MR, &mr) < 0)
			goto fail;
		layer_mr_ids[i] = mr.mr_id;
	}
	layers_allocated = n_layers;
	return 0;

fail:
	/* Cleanup partial allocation */
	for (int j = 0; j < i; j++) {
		ioctl(dev_fd, IOCTL_DEREGISTER_MR, &layer_mr_ids[j]);
		ioctl(dev_fd, IOCTL_DESTROY_BUFFER, &layer_buf_ids[j]);
	}
	layers_allocated = 0;
	return -1;
}

static void free_layers(void)
{
	for (int i = 0; i < layers_allocated; i++) {
		ioctl(dev_fd, IOCTL_DEREGISTER_MR, &layer_mr_ids[i]);
		ioctl(dev_fd, IOCTL_DESTROY_BUFFER, &layer_buf_ids[i]);
	}
	layers_allocated = 0;
}

/* ── Simulation passes ── */

static double run_forward_sequential(int n_layers, __u32 shard_bytes, int iters)
{
	uint64_t total = 0;
	for (int i = 0; i < n_layers; i++) {
		struct dmaplane_bench_params p = {
			.mr_id      = layer_mr_ids[i],
			.msg_size   = shard_bytes,
			.iterations = (__u32)iters,
			.queue_depth = 1,
		};
		if (ioctl(dev_fd, IOCTL_STREAMING_BENCH, &p) < 0)
			return -1;
		total += p.total_ns;
	}
	return (double)total;
}

static double run_forward_pipelined(int n_layers, __u32 shard_bytes,
				    int iters, int qdepth)
{
	struct dmaplane_bench_params p = {
		.mr_id      = layer_mr_ids[0],
		.msg_size   = shard_bytes,
		.iterations = (__u32)(n_layers * iters),
		.queue_depth = (__u32)qdepth,
	};
	if (ioctl(dev_fd, IOCTL_STREAMING_BENCH, &p) < 0)
		return -1;
	return (double)p.total_ns;
}

static double run_backward(int n_layers, __u32 shard_bytes, int iters)
{
	uint64_t total = 0;
	for (int i = n_layers - 1; i >= 0; i--) {
		struct dmaplane_bench_params p = {
			.mr_id      = layer_mr_ids[i],
			.msg_size   = shard_bytes,
			.iterations = (__u32)iters,
		};
		if (ioctl(dev_fd, IOCTL_PINGPONG_BENCH, &p) < 0)
			return -1;
		total += p.total_ns;
	}
	return (double)total;
}

/* ── Run one epoch ── */

static int run_epoch(void)
{
	int n_layers    = params[P_LAYERS].value;
	int shard_kb    = params[P_SHARD_KB].value;
	int qdepth      = params[P_QDEPTH].value;
	int iters       = params[P_ITERATIONS].value;
	__u32 shard_bytes = (__u32)shard_kb * 1024;
	double total_data = (double)shard_bytes * n_layers * iters;

	snprintf(status_msg, sizeof(status_msg),
		 "Running epoch %d: %d layers x %d KB x QD=%d ...",
		 epoch_count + 1, n_layers, shard_kb, qdepth);
	refresh();

	/* Allocate fresh resources */
	teardown_rdma();
	if (setup_rdma() < 0) {
		snprintf(status_msg, sizeof(status_msg),
			 "RDMA setup failed: %s", strerror(errno));
		return -1;
	}
	if (allocate_layers(n_layers, shard_bytes) < 0) {
		snprintf(status_msg, sizeof(status_msg),
			 "Layer allocation failed: %s", strerror(errno));
		return -1;
	}

	/* Forward sequential */
	double fwd_seq_ns = run_forward_sequential(n_layers, shard_bytes, iters);
	if (fwd_seq_ns < 0) {
		snprintf(status_msg, sizeof(status_msg),
			 "Sequential forward failed: %s", strerror(errno));
		free_layers();
		return -1;
	}

	/* Forward pipelined */
	double fwd_pipe_ns = run_forward_pipelined(n_layers, shard_bytes, iters, qdepth);
	if (fwd_pipe_ns < 0) {
		snprintf(status_msg, sizeof(status_msg),
			 "Pipelined forward failed: %s", strerror(errno));
		free_layers();
		return -1;
	}

	/* Backward */
	double bwd_ns = run_backward(n_layers, shard_bytes, iters);
	if (bwd_ns < 0) {
		snprintf(status_msg, sizeof(status_msg),
			 "Backward pass failed: %s", strerror(errno));
		free_layers();
		return -1;
	}

	free_layers();

	/* Record result */
	epoch_count++;
	struct epoch_result *r;
	if (history_count < MAX_HISTORY) {
		r = &history[history_count++];
	} else {
		/* Shift history */
		memmove(&history[0], &history[1],
			(MAX_HISTORY - 1) * sizeof(history[0]));
		r = &history[MAX_HISTORY - 1];
	}

	r->epoch_num    = epoch_count;
	r->layers       = n_layers;
	r->shard_kb     = shard_kb;
	r->qdepth       = qdepth;
	r->fwd_seq_ms   = fwd_seq_ns / 1e6;
	r->fwd_seq_mbps = (fwd_seq_ns > 0) ? total_data * 1000.0 / fwd_seq_ns : 0;
	r->fwd_pipe_ms  = fwd_pipe_ns / 1e6;
	r->fwd_pipe_mbps = (fwd_pipe_ns > 0) ? total_data * 1000.0 / fwd_pipe_ns : 0;
	r->speedup      = (fwd_pipe_ns > 0) ? fwd_seq_ns / fwd_pipe_ns : 0;
	r->bwd_ms       = bwd_ns / 1e6;
	r->bwd_mbps     = (bwd_ns > 0) ? total_data * 1000.0 / bwd_ns : 0;
	r->total_ms     = r->fwd_pipe_ms + r->bwd_ms;

	snprintf(status_msg, sizeof(status_msg),
		 "Epoch %d done: %.1fx speedup, %.0f MB/s pipelined",
		 epoch_count, r->speedup, r->fwd_pipe_mbps);

	return 0;
}

/* ── Drawing ── */

static void draw_header(void)
{
	attron(A_BOLD);
	mvprintw(0, 0, " Accelerator Weight Streaming Simulator");
	attroff(A_BOLD);
	mvprintw(0, 50, "IB: %s", ib_dev_name);
}

static void draw_params(int start_y)
{
	mvprintw(start_y, 1, "Parameters:");

	for (int i = 0; i < P_COUNT; i++) {
		int y = start_y + 1 + i;
		if (i == selected_param) {
			attron(A_REVERSE);
		}

		mvprintw(y, 2, "  %-20s", params[i].name);

		/* Draw slider bar */
		mvprintw(y, 24, "[");
		int bar_width = 20;
		int range;
		int fill;

		if (params[i].step == 0) {
			/* Power-of-2: count steps */
			int steps = 0, v = params[i].min;
			while (v < params[i].max) { v *= 2; steps++; }
			int cur = 0; v = params[i].min;
			while (v < params[i].value) { v *= 2; cur++; }
			fill = (steps > 0) ? (cur * bar_width / steps) : 0;
		} else {
			range = params[i].max - params[i].min;
			fill = (range > 0) ? ((params[i].value - params[i].min) * bar_width / range) : 0;
		}

		for (int j = 0; j < bar_width; j++) {
			if (j < fill)
				addch(ACS_CKBOARD);
			else
				addch(' ');
		}
		printw("]");

		/* Value */
		printw(" %4d %s", params[i].value, params[i].unit);

		if (i == selected_param)
			attroff(A_REVERSE);
	}

	/* Derived info */
	int total_kb = params[P_LAYERS].value * params[P_SHARD_KB].value;
	mvprintw(start_y + P_COUNT + 2, 2, "  Total model size: %d KB (%d layers x %d KB)",
		 total_kb, params[P_LAYERS].value, params[P_SHARD_KB].value);
}

static void draw_progress_bar(int y, const char *label, double ms, double mbps, int width)
{
	mvprintw(y, 2, "%-12s", label);

	/* Scale bar to max observed */
	double max_ms = 1;
	for (int i = 0; i < history_count; i++) {
		if (history[i].fwd_seq_ms > max_ms) max_ms = history[i].fwd_seq_ms;
		if (history[i].bwd_ms > max_ms) max_ms = history[i].bwd_ms;
	}

	int bar_len = (max_ms > 0) ? (int)(ms / max_ms * width) : 0;
	if (bar_len > width) bar_len = width;
	if (bar_len < 1 && ms > 0) bar_len = 1;

	printw(" ");
	attron(COLOR_PAIR(1));
	for (int j = 0; j < bar_len; j++)
		addch(ACS_CKBOARD);
	attroff(COLOR_PAIR(1));
	for (int j = bar_len; j < width; j++)
		addch(' ');

	printw(" %7.1f ms  %5.0f MB/s", ms, mbps);
}

static void draw_results(int start_y)
{
	int rows, cols;
	getmaxyx(stdscr, rows, cols);
	(void)cols;

	mvprintw(start_y, 1, "Results:");

	if (history_count == 0) {
		mvprintw(start_y + 1, 3, "(no runs yet)");
		return;
	}

	/* Show latest result as bar chart */
	struct epoch_result *r = &history[history_count - 1];

	mvprintw(start_y + 1, 2, "Epoch %d: %d layers x %d KB, QD=%d",
		 r->epoch_num, r->layers, r->shard_kb, r->qdepth);

	draw_progress_bar(start_y + 3, "Fwd (seq)",  r->fwd_seq_ms,  r->fwd_seq_mbps, 25);
	draw_progress_bar(start_y + 4, "Fwd (pipe)", r->fwd_pipe_ms, r->fwd_pipe_mbps, 25);
	draw_progress_bar(start_y + 5, "Backward",   r->bwd_ms,      r->bwd_mbps, 25);

	mvprintw(start_y + 7, 2, "Pipeline speedup: ");
	attron(A_BOLD);
	printw("%.1fx", r->speedup);
	attroff(A_BOLD);
	printw("   Total epoch: %.1f ms", r->total_ms);

	/* History table */
	int table_y = start_y + 9;
	int avail = rows - table_y - 3;
	if (avail < 1) return;

	attron(A_UNDERLINE);
	mvprintw(table_y, 2,
		 " Ep  Layers  Shard  QD  Seq(ms) Pipe(ms) Spdup  Bwd(ms)  Total");
	attroff(A_UNDERLINE);

	int show = history_count;
	if (show > avail) show = avail;
	int start_idx = history_count - show;

	for (int i = 0; i < show; i++) {
		struct epoch_result *h = &history[start_idx + i];
		mvprintw(table_y + 1 + i, 2,
			 "%3d  %5d  %4dKB  %2d  %7.1f  %7.1f  %4.1fx  %7.1f  %7.1f",
			 h->epoch_num, h->layers, h->shard_kb, h->qdepth,
			 h->fwd_seq_ms, h->fwd_pipe_ms, h->speedup,
			 h->bwd_ms, h->total_ms);
	}
}

static void draw_status(void)
{
	int rows, cols;
	getmaxyx(stdscr, rows, cols);
	(void)cols;

	/* Controls */
	mvprintw(rows - 2, 1, " ");
	attron(A_DIM);
	printw("Up/Dn");
	attroff(A_DIM);
	printw(":select ");
	attron(A_DIM);
	printw("Lt/Rt");
	attroff(A_DIM);
	printw(":adjust ");
	attron(A_DIM);
	printw("Enter");
	attroff(A_DIM);
	printw(":run ");
	attron(A_DIM);
	printw("c");
	attroff(A_DIM);
	printw(":continuous ");
	attron(A_DIM);
	printw("s");
	attroff(A_DIM);
	printw(":stop ");
	attron(A_DIM);
	printw("q");
	attroff(A_DIM);
	printw(":quit");

	/* Status line */
	attron(A_REVERSE);
	mvprintw(rows - 1, 0, " %-*s", cols - 1, status_msg);
	attroff(A_REVERSE);
}

static void draw_screen(void)
{
	erase();
	draw_header();

	int y = 2;
	mvhline(1, 0, ACS_HLINE, COLS);

	draw_params(y);
	y += P_COUNT + 4;

	mvhline(y, 0, ACS_HLINE, COLS);
	y++;

	draw_results(y);
	draw_status();

	refresh();
}

/* ── Main ── */

static volatile sig_atomic_t got_signal = 0;

static void sig_handler(int sig)
{
	(void)sig;
	got_signal = 1;
}

int main(int argc, char *argv[])
{
	if (argc > 1) {
		snprintf(ib_dev_name, sizeof(ib_dev_name), "%s", argv[1]);
	} else {
		pick_default_ib_device();
	}

	if (!ib_dev_name[0]) {
		fprintf(stderr, "No IB device specified and none auto-detected.\n");
		fprintf(stderr, "Usage: sudo %s <ib_device>\n", argv[0]);
		return 1;
	}

	/* Open device */
	dev_fd = open(DEV_PATH, O_RDWR);
	if (dev_fd < 0) {
		fprintf(stderr, "Cannot open %s: %s\n", DEV_PATH, strerror(errno));
		fprintf(stderr, "Is dmaplane.ko loaded?\n");
		return 1;
	}

	/* Initial RDMA setup */
	if (setup_rdma() < 0) {
		fprintf(stderr, "RDMA setup failed for device '%s': %s\n",
			ib_dev_name, strerror(errno));
		print_available_ib_devices(stderr);
		close(dev_fd);
		return 1;
	}

	/* Signal handling for clean exit */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Init ncurses */
	initscr();
	cbreak();
	noecho();
	keypad(stdscr, TRUE);
	curs_set(0);
	timeout(100); /* 100ms for continuous mode polling */

	if (has_colors()) {
		start_color();
		init_pair(1, COLOR_GREEN, COLOR_BLACK);
		init_pair(2, COLOR_CYAN, COLOR_BLACK);
		init_pair(3, COLOR_YELLOW, COLOR_BLACK);
	}

	draw_screen();

	while (!got_signal) {
		int ch = getch();

		if (ch == 'q' || ch == 'Q')
			break;

		switch (ch) {
		case KEY_UP:
			selected_param--;
			if (selected_param < 0) selected_param = P_COUNT - 1;
			break;

		case KEY_DOWN:
			selected_param++;
			if (selected_param >= P_COUNT) selected_param = 0;
			break;

		case KEY_LEFT:
			adjust_param(selected_param, -1);
			break;

		case KEY_RIGHT:
			adjust_param(selected_param, 1);
			break;

		case '\n':
		case 'r':
		case 'R':
			running = 1;
			continuous = 0;
			run_epoch();
			running = 0;
			break;

		case 'c':
		case 'C':
			continuous = 1;
			snprintf(status_msg, sizeof(status_msg),
				 "Continuous mode — press 's' to stop");
			break;

		case 's':
		case 'S':
			continuous = 0;
			snprintf(status_msg, sizeof(status_msg),
				 "Stopped. Press Enter to run, 'c' for continuous");
			break;

		default:
			break;
		}

		/* Continuous mode — run an epoch each cycle */
		if (continuous && !running) {
			running = 1;
			run_epoch();
			running = 0;

			/* Auto-cycle parameters: bump shard size each epoch */
			/* (can be changed to cycle other params) */
		}

		draw_screen();
	}

	/* Cleanup */
	endwin();

	teardown_rdma();
	close(dev_fd);

	printf("Weight streamer TUI exited cleanly.\n");
	printf("Ran %d epochs total.\n", epoch_count);
	return 0;
}
