// SPDX-License-Identifier: GPL-2.0
/*
 * trace_explorer.c — Instrumentation & tracepoint demo
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Demonstrates Phase 7 instrumentation features:
 *   1. Enables a kernel tracepoint via ftrace sysfs
 *   2. Triggers the tracepoint by creating a buffer
 *   3. Reads the trace buffer to show captured events
 *   4. Runs RDMA pingpong (if rxe available) to populate histogram
 *   5. Reads the latency histogram via ioctl
 *   6. Reads debugfs files for live driver state
 *
 * Requires: dmaplane.ko loaded, root privileges.
 * Optional: Soft-RoCE for RDMA histogram demo.
 *
 * Usage: sudo ./examples/misc/trace_explorer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <stdint.h>

#include "dmaplane_uapi.h"

#define DEV_PATH	"/dev/dmaplane"

/* IB access flags */
#define IB_ACCESS_LOCAL_WRITE  (1)
#define IB_ACCESS_REMOTE_WRITE (1 << 1)
#define IB_ACCESS_REMOTE_READ  (1 << 2)

/* ── Helpers ───────────────────────────────────────────────── */

static int trace_enable(const char *event, int on)
{
	char path[256];
	int fd;

	snprintf(path, sizeof(path),
		 "/sys/kernel/debug/tracing/events/dmaplane/%s/enable",
		 event);
	fd = open(path, O_WRONLY);
	if (fd < 0)
		return -1;
	if (write(fd, on ? "1" : "0", 1) < 0) {
		close(fd);
		return -1;
	}
	close(fd);
	return 0;
}

static void trace_clear(void)
{
	int fd = open("/sys/kernel/debug/tracing/trace", O_WRONLY | O_TRUNC);

	if (fd >= 0)
		close(fd);
}

static void trace_show(const char *filter)
{
	char buf[8192];
	int fd;
	ssize_t n;
	int found = 0;

	fd = open("/sys/kernel/debug/tracing/trace", O_RDONLY);
	if (fd < 0) {
		printf("  (cannot read trace buffer: %s)\n", strerror(errno));
		return;
	}

	while ((n = read(fd, buf, sizeof(buf) - 1)) > 0) {
		char *line, *save;

		buf[n] = '\0';
		for (line = strtok_r(buf, "\n", &save); line;
		     line = strtok_r(NULL, "\n", &save)) {
			if (strstr(line, filter)) {
				printf("  %s\n", line);
				found++;
			}
		}
	}
	close(fd);

	if (!found)
		printf("  (no events matching '%s')\n", filter);
}

static ssize_t read_debugfs(const char *path, char *buf, size_t bufsz)
{
	int fd;
	ssize_t total = 0, n;

	fd = open(path, O_RDONLY);
	if (fd < 0)
		return -1;

	while (total < (ssize_t)(bufsz - 1)) {
		n = read(fd, buf + total, bufsz - 1 - total);
		if (n <= 0)
			break;
		total += n;
	}
	buf[total] = '\0';
	close(fd);
	return total;
}

static int find_rxe_device(char *name, size_t len)
{
	DIR *dir;
	struct dirent *ent;

	dir = opendir("/sys/class/infiniband");
	if (!dir)
		return -1;
	while ((ent = readdir(dir)) != NULL) {
		if (strncmp(ent->d_name, "rxe_", 4) == 0 &&
		    strlen(ent->d_name) < len) {
			memcpy(name, ent->d_name, strlen(ent->d_name) + 1);
			closedir(dir);
			return 0;
		}
	}
	closedir(dir);
	return -1;
}

/* ── Main ─────────────────────────────────────────────────── */

int main(void)
{
	int fd;
	char rxe_name[64];
	char dbg[8192];
	__u32 buf_id = 0, mr_id = 0;
	int rdma_up = 0;

	printf("dmaplane Trace Explorer\n");
	printf("=======================\n");

	fd = open(DEV_PATH, O_RDWR);
	if (fd < 0) {
		perror("open " DEV_PATH);
		return 1;
	}

	/* ── Step 1: Show available tracepoints ── */
	printf("\n--- Available dmaplane tracepoints ---\n");
	{
		ssize_t n = read_debugfs(
			"/sys/kernel/debug/tracing/available_events",
			dbg, sizeof(dbg));
		if (n > 0) {
			char *line, *save;

			for (line = strtok_r(dbg, "\n", &save); line;
			     line = strtok_r(NULL, "\n", &save)) {
				if (strncmp(line, "dmaplane:", 9) == 0)
					printf("  %s\n", line);
			}
		} else {
			printf("  (tracing not accessible)\n");
		}
	}

	/* ── Step 2: Enable tracepoint, create buffer, capture event ── */
	printf("\n--- Tracepoint capture: dmaplane_buf_alloc ---\n");

	trace_clear();

	if (trace_enable("dmaplane_buf_alloc", 1) < 0) {
		printf("  (cannot enable tracepoint — ftrace not accessible)\n");
	} else {
		struct dmaplane_buf_params bp = {
			.alloc_type = DMAPLANE_BUF_TYPE_PAGES,
			.size = 65536,
			.numa_node = DMAPLANE_NUMA_ANY,
		};

		printf("  Enabled dmaplane_buf_alloc tracepoint\n");
		printf("  Creating 64 KB page-backed buffer...\n");

		if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &bp) == 0) {
			buf_id = bp.buf_id;
			printf("  Created buf_id=%u\n", buf_id);
		} else {
			printf("  create_buffer failed: %s\n", strerror(errno));
		}

		/* Small delay for trace buffer to be written */
		usleep(1000);

		printf("\n  Captured trace events:\n");
		trace_show("dmaplane_buf_alloc");

		trace_enable("dmaplane_buf_alloc", 0);
		printf("\n  Disabled tracepoint\n");

		/* Destroy the demo buffer */
		if (buf_id) {
			ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf_id);
			buf_id = 0;
		}
	}

	/* ── Step 3: RDMA pingpong + histogram (if rxe available) ── */
	printf("\n--- Latency histogram ---\n");

	if (find_rxe_device(rxe_name, sizeof(rxe_name)) == 0) {
		struct dmaplane_rdma_setup rs = {0};
		struct dmaplane_buf_params bp = {
			.alloc_type = DMAPLANE_BUF_TYPE_PAGES,
			.size = 1048576,
			.numa_node = DMAPLANE_NUMA_ANY,
		};
		struct dmaplane_mr_params mp = {0};
		struct dmaplane_bench_params bench = {0};
		struct dmaplane_hist_params hp = {0};

		snprintf(rs.ib_dev_name, sizeof(rs.ib_dev_name), "%s", rxe_name);
		rs.port = 1;

		if (ioctl(fd, DMAPLANE_IOCTL_SETUP_RDMA, &rs) == 0 &&
		    rs.status == 0) {
			rdma_up = 1;
			printf("  RDMA initialized (%s)\n", rxe_name);
		}

		if (rdma_up &&
		    ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &bp) == 0) {
			buf_id = bp.buf_id;

			mp.buf_id = buf_id;
			mp.access_flags = IB_ACCESS_LOCAL_WRITE |
					  IB_ACCESS_REMOTE_WRITE |
					  IB_ACCESS_REMOTE_READ;

			if (ioctl(fd, DMAPLANE_IOCTL_REGISTER_MR, &mp) == 0) {
				mr_id = mp.mr_id;

				/* Reset histogram for clean measurement */
				hp.reset = 1;
				ioctl(fd, DMAPLANE_IOCTL_GET_HISTOGRAM, &hp);

				/* Run 500-iteration pingpong */
				bench.mr_id = mr_id;
				bench.msg_size = 4096;
				bench.iterations = 500;

				printf("  Running pingpong: 500 x 4 KB...\n");
				if (ioctl(fd, DMAPLANE_IOCTL_PINGPONG_BENCH,
					  &bench) == 0) {
					printf("  Avg latency: %llu ns, "
					       "P99: %llu ns, "
					       "Throughput: %llu MB/s\n",
					       (unsigned long long)bench.avg_latency_ns,
					       (unsigned long long)bench.p99_latency_ns,
					       (unsigned long long)bench.throughput_mbps);
				}
			}
		}

		/* Read histogram */
		memset(&hp, 0, sizeof(hp));
		if (ioctl(fd, DMAPLANE_IOCTL_GET_HISTOGRAM, &hp) == 0 &&
		    hp.count > 0) {
			int i;

			printf("\n  Histogram: %llu samples\n",
			       (unsigned long long)hp.count);
			printf("  avg=%llu ns  min=%llu ns  max=%llu ns\n",
			       (unsigned long long)hp.avg_ns,
			       (unsigned long long)hp.min_ns,
			       (unsigned long long)hp.max_ns);
			printf("  P50=%llu ns  P99=%llu ns  P999=%llu ns\n",
			       (unsigned long long)hp.p50_ns,
			       (unsigned long long)hp.p99_ns,
			       (unsigned long long)hp.p999_ns);
			printf("\n  %-4s  %-16s  %8s  %6s\n",
			       "Bin", "Range (us)", "Count", "Pct");
			for (i = 0; i < DMAPLANE_HIST_BUCKETS; i++) {
				unsigned int lo, hi;

				if (hp.buckets[i] == 0)
					continue;
				lo = (i == 0) ? 0 : (1u << i);
				hi = (i < DMAPLANE_HIST_BUCKETS - 1)
					? (1u << (i + 1)) : 0;
				if (hi > 0)
					printf("  %-4d  [%5u, %5u)      %8llu  %5.1f%%\n",
					       i, lo, hi,
					       (unsigned long long)hp.buckets[i],
					       hp.count > 0
						? 100.0 * hp.buckets[i] / hp.count
						: 0.0);
				else
					printf("  %-4d  [%5u,    +inf)    %8llu  %5.1f%%\n",
					       i, lo,
					       (unsigned long long)hp.buckets[i],
					       hp.count > 0
						? 100.0 * hp.buckets[i] / hp.count
						: 0.0);
			}
		} else {
			printf("  (no histogram samples — RDMA benchmarks populate the histogram)\n");
		}
	} else {
		printf("  No rxe device found — skipping RDMA histogram demo\n");
		printf("  (run: rdma link add rxe_<iface> type rxe netdev <iface>)\n");
	}

	/* ── Step 4: Show debugfs state ── */
	printf("\n--- debugfs: /sys/kernel/debug/dmaplane/stats ---\n");
	{
		ssize_t n = read_debugfs(
			"/sys/kernel/debug/dmaplane/stats", dbg, sizeof(dbg));
		if (n > 0)
			printf("%s", dbg);
		else
			printf("  (debugfs not accessible)\n");
	}

	/* ── Cleanup ── */
	if (mr_id)
		ioctl(fd, DMAPLANE_IOCTL_DEREGISTER_MR, &mr_id);
	if (buf_id)
		ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf_id);
	if (rdma_up)
		ioctl(fd, DMAPLANE_IOCTL_TEARDOWN_RDMA);

	close(fd);
	printf("\nDone.\n");
	return 0;
}
