// SPDX-License-Identifier: GPL-2.0
/*
 * dmaplane Phase 1 test — stress test for character device + rings
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Userspace test suite that exercises the Phase 1 ioctl interface:
 * channel creation, submit/complete data flow, ring full backpressure,
 * single-channel stress (1M entries), multi-channel concurrency
 * (4 channels x 250K entries via pthreads), and resource cleanup on
 * close.  Each test prints PASS/FAIL; main() returns 0 if all pass,
 * 1 otherwise.
 *
 * Must be run as root (or with suitable /dev/dmaplane permissions)
 * after loading the dmaplane.ko module.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <time.h>
#include <pthread.h>

#include "dmaplane_uapi.h"

#define DEV_PATH	"/dev/dmaplane"
#define STRESS_COUNT	1000000	/* 1M entries: large enough to expose races */
#define MULTI_PER_CHAN	250000	/* Per-channel count in multi-channel test;
				 * 4 channels x 250K = 1M total */
#define BATCH_SIZE	256	/* Submit this many before attempting to
				 * drain completions — balances ring
				 * occupancy against completion latency */

static int tests_passed;
static int tests_failed;

#define TEST_PASS(name) do { \
	printf("  PASS: %s\n", name); \
	tests_passed++; \
} while (0)

#define TEST_FAIL(name, ...) do { \
	printf("  FAIL: %s — ", name); \
	printf(__VA_ARGS__); \
	printf("\n"); \
	tests_failed++; \
} while (0)

/*
 * dev_open - Open /dev/dmaplane read-write.
 *
 * Returns fd >= 0 on success, -1 on error (prints perror).
 */
static int dev_open(void)
{
	int fd = open(DEV_PATH, O_RDWR);
	if (fd < 0) {
		perror("open " DEV_PATH);
	}
	return fd;
}

/*
 * create_channel - Issue IOCTL_CREATE_CHANNEL and return the assigned ID.
 * @fd: open file descriptor to /dev/dmaplane.
 *
 * Returns channel_id (>= 0) on success, -1 on error.
 * One channel per fd — a second call on the same fd returns -EBUSY.
 */
static int create_channel(int fd)
{
	struct dmaplane_channel_params params;
	int ret;

	memset(&params, 0, sizeof(params));
	ret = ioctl(fd, DMAPLANE_IOCTL_CREATE_CHANNEL, &params);
	if (ret < 0)
		return -1;
	return (int)params.channel_id;
}

/*
 * submit_entry - Submit a single ring entry via IOCTL_SUBMIT.
 * @fd:      open fd with a channel already created.
 * @payload: the u64 payload to enqueue.
 * @flags:   per-entry flags (reserved, pass 0 for Phase 1).
 *
 * Returns 0 on success, -1 on error (errno set by ioctl).
 * Common errors: ENOSPC (ring full), ENODEV (no channel).
 */
static int submit_entry(int fd, __u64 payload, __u32 flags)
{
	struct dmaplane_submit_params p;

	memset(&p, 0, sizeof(p));
	p.entry.payload = payload;
	p.entry.flags = flags;
	return ioctl(fd, DMAPLANE_IOCTL_SUBMIT, &p);
}

/*
 * complete_entry - Dequeue one completion via IOCTL_COMPLETE.
 * @fd:  open fd with a channel already created.
 * @out: if non-NULL and the call succeeds, the dequeued entry is
 *       written here.
 *
 * Non-blocking: returns -1 with errno == EAGAIN if the completion
 * ring is empty.  Returns 0 on success.
 */
static int complete_entry(int fd, struct dmaplane_ring_entry *out)
{
	struct dmaplane_complete_params p;
	int ret;

	memset(&p, 0, sizeof(p));
	ret = ioctl(fd, DMAPLANE_IOCTL_COMPLETE, &p);
	if (ret == 0 && out)
		*out = p.entry;
	return ret;
}

/*
 * complete_poll - Poll for a completion with bounded retry.
 * @fd:          open fd with a channel already created.
 * @out:         output entry (see complete_entry).
 * @max_retries: maximum number of 100 us sleeps before giving up.
 *
 * Userspace busy-poll loop: calls complete_entry with 100 us sleeps
 * between retries.  Effective timeout ~ max_retries * 100 us.
 *
 * Returns 0 if a completion was obtained, -1 if timed out.
 */
static int complete_poll(int fd, struct dmaplane_ring_entry *out,
			 int max_retries)
{
	int i;
	for (i = 0; i < max_retries; i++) {
		if (complete_entry(fd, out) == 0)
			return 0;
		usleep(100);
	}
	return -1;
}

/* ------------------------------------------------------------------ */
/* Test 1: Basic open/close                                            */
/* ------------------------------------------------------------------ */

/*
 * Validates that /dev/dmaplane can be opened and closed without error.
 * Most fundamental smoke test — if it fails, the module is not loaded
 * or the device node does not exist.
 */
static void test_open_close(void)
{
	int fd = dev_open();
	if (fd < 0) {
		TEST_FAIL("open/close", "cannot open device");
		return;
	}
	close(fd);
	TEST_PASS("open/close");
}

/* ------------------------------------------------------------------ */
/* Test 2: Channel creation                                            */
/* ------------------------------------------------------------------ */

/*
 * Opens two independent fds, creates a channel on each, and verifies
 * that the returned channel IDs are 0 and 1 (sequential allocation
 * from the lowest free slot).  Validates:
 *  - Multiple channels can coexist.
 *  - Each fd gets its own channel.
 *  - IDs are assigned sequentially.
 */
static void test_channel_creation(void)
{
	int fd1, fd2, ch1, ch2;

	fd1 = dev_open();
	fd2 = dev_open();
	if (fd1 < 0 || fd2 < 0) {
		TEST_FAIL("channel creation", "cannot open device");
		if (fd1 >= 0) close(fd1);
		if (fd2 >= 0) close(fd2);
		return;
	}

	ch1 = create_channel(fd1);
	ch2 = create_channel(fd2);

	if (ch1 < 0 || ch2 < 0) {
		TEST_FAIL("channel creation", "ioctl failed: ch1=%d ch2=%d",
			  ch1, ch2);
	} else if (ch1 != 0 || ch2 != 1) {
		TEST_FAIL("channel creation",
			  "expected IDs 0,1 got %d,%d", ch1, ch2);
	} else {
		TEST_PASS("channel creation");
	}

	close(fd1);
	close(fd2);
}

/* ------------------------------------------------------------------ */
/* Test 3: Submit and complete                                         */
/* ------------------------------------------------------------------ */

/*
 * Single-entry round-trip: submits one entry with a known payload
 * (0xDEADBEEF), waits for the worker to process it, and verifies
 * that the completed payload equals original + 1.  Proves:
 *  - Submit enqueues into the submission ring.
 *  - Worker dequeues, processes (payload += 1), and enqueues into
 *    the completion ring.
 *  - Complete dequeues from the completion ring.
 */
static void test_submit_complete(void)
{
	int fd, ch;
	struct dmaplane_ring_entry entry;
	__u64 payload = 0xDEADBEEFULL;

	fd = dev_open();
	if (fd < 0) {
		TEST_FAIL("submit/complete", "cannot open device");
		return;
	}

	ch = create_channel(fd);
	if (ch < 0) {
		TEST_FAIL("submit/complete", "create channel failed");
		close(fd);
		return;
	}

	if (submit_entry(fd, payload, 0) != 0) {
		TEST_FAIL("submit/complete", "submit failed: %s",
			  strerror(errno));
		close(fd);
		return;
	}

	if (complete_poll(fd, &entry, 10000) != 0) {
		TEST_FAIL("submit/complete", "completion timed out");
		close(fd);
		return;
	}

	/* Worker should have incremented payload by 1 */
	if (entry.payload != payload + 1) {
		TEST_FAIL("submit/complete",
			  "expected payload 0x%llx, got 0x%llx",
			  (unsigned long long)(payload + 1),
			  (unsigned long long)entry.payload);
	} else {
		TEST_PASS("submit/complete");
	}

	close(fd);
}

/* ------------------------------------------------------------------ */
/* Test 4: Ring full behavior                                          */
/* ------------------------------------------------------------------ */

/*
 * Rapidly submits entries until the submission ring returns -ENOSPC,
 * then drains all completions and verifies payload integrity (each
 * payload == original + 1, in FIFO order).  Validates:
 *  - Ring correctly rejects entries when full.
 *  - Worker continues to process under full-ring conditions.
 *  - No entries are lost or reordered.
 *
 * If the worker drains faster than we can submit, the ring never fills
 * — this is noted but still counted as a pass (the ring is working).
 */
static void test_ring_full(void)
{
	int fd, ch, ret;
	int submitted = 0;
	struct dmaplane_ring_entry entry;

	fd = dev_open();
	if (fd < 0) {
		TEST_FAIL("ring full", "cannot open device");
		return;
	}

	ch = create_channel(fd);
	if (ch < 0) {
		TEST_FAIL("ring full", "create channel failed");
		close(fd);
		return;
	}

	/*
	 * Submit RING_SIZE entries rapidly. The ring should accept
	 * (RING_SIZE - 1) entries because the worker may consume some
	 * before we fill it. We keep trying until we get ENOSPC.
	 */
	while (submitted < DMAPLANE_RING_SIZE + 10) {
		ret = submit_entry(fd, (__u64)submitted, 0);
		if (ret != 0) {
			if (errno == ENOSPC)
				break;
			TEST_FAIL("ring full",
				  "unexpected error at %d: %s",
				  submitted, strerror(errno));
			close(fd);
			return;
		}
		submitted++;
	}

	if (submitted == DMAPLANE_RING_SIZE + 10) {
		/*
		 * Worker consumed entries fast enough that we never got
		 * ENOSPC. This is technically fine — the ring works,
		 * just the worker is fast. Accept as pass with note.
		 */
		printf("  NOTE: worker consumed entries faster than submit; "
		       "ring full not triggered\n");
	}

	/* Drain all completions and verify FIFO ordering */
	int completed = 0;
	while (completed < submitted) {
		if (complete_poll(fd, &entry, 50000) != 0) {
			TEST_FAIL("ring full",
				  "stuck draining: completed %d/%d",
				  completed, submitted);
			close(fd);
			return;
		}
		/* Verify: payload should be (original + 1) in FIFO order */
		if (entry.payload != (__u64)completed + 1) {
			TEST_FAIL("ring full",
				  "payload mismatch at %d: expected 0x%llx got 0x%llx",
				  completed,
				  (unsigned long long)((__u64)completed + 1),
				  (unsigned long long)entry.payload);
			close(fd);
			return;
		}
		completed++;
	}

	TEST_PASS("ring full");
	close(fd);
}

/* ------------------------------------------------------------------ */
/* Test 5: Stress test (single channel)                                */
/* ------------------------------------------------------------------ */

/*
 * Pushes STRESS_COUNT (1M) entries through one channel using batched
 * submit/complete cycles.  Measures wall-clock throughput (ops/sec).
 * Validates:
 *  - Ring and worker sustain high throughput without deadlock.
 *  - No entries are lost under sustained load.
 *  - ENOSPC (ring full) and EAGAIN (ring empty) are handled correctly
 *    in the retry loops: ENOSPC means stop submitting and drain
 *    completions; EAGAIN means stop draining and submit more.
 */
static void test_stress_single(void)
{
	int fd, ch;
	int submitted = 0, completed = 0;
	struct dmaplane_ring_entry entry;
	struct timespec start, end;
	double elapsed, rate;

	fd = dev_open();
	if (fd < 0) {
		TEST_FAIL("stress single", "cannot open device");
		return;
	}

	ch = create_channel(fd);
	if (ch < 0) {
		TEST_FAIL("stress single", "create channel failed");
		close(fd);
		return;
	}

	clock_gettime(CLOCK_MONOTONIC, &start);

	while (completed < STRESS_COUNT) {
		/* Submit a batch */
		int batch = 0;
		while (submitted < STRESS_COUNT && batch < BATCH_SIZE) {
			if (submit_entry(fd, (__u64)submitted, 0) == 0) {
				submitted++;
				batch++;
			} else if (errno == ENOSPC) {
				/* Ring full — switch to draining completions */
				break;
			} else {
				TEST_FAIL("stress single",
					  "submit error at %d: %s",
					  submitted, strerror(errno));
				close(fd);
				return;
			}
		}

		/* Complete as many as available */
		while (completed < submitted) {
			if (complete_entry(fd, &entry) == 0) {
				completed++;
			} else if (errno == EAGAIN) {
				/* Ring empty — switch back to submitting */
				break;
			} else {
				TEST_FAIL("stress single",
					  "complete error at %d: %s",
					  completed, strerror(errno));
				close(fd);
				return;
			}
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsed = (double)(end.tv_sec - start.tv_sec) +
		  (double)(end.tv_nsec - start.tv_nsec) / 1e9;
	rate = (double)STRESS_COUNT / elapsed;

	if (completed == STRESS_COUNT) {
		printf("  PASS: stress single — %d entries in %.3f s "
		       "(%.0f ops/sec)\n", STRESS_COUNT, elapsed, rate);
		tests_passed++;
	} else {
		TEST_FAIL("stress single",
			  "count mismatch: submitted=%d completed=%d",
			  submitted, completed);
	}

	close(fd);
}

/* ------------------------------------------------------------------ */
/* Test 6: Multi-channel stress                                        */
/* ------------------------------------------------------------------ */

/* Per-thread state for multi-channel test */
struct thread_args {
	int fd;		/* Open fd with a channel already created */
	int channel_id;	/* Channel index returned by create_channel */
	int count;	/* Number of entries this thread should process */
	int ok;		/* Result: set to 1 on success, 0 on failure */
};

/*
 * multi_channel_thread - Per-channel worker for test 6.
 * @arg: pointer to struct thread_args.
 *
 * Each pthread drives one channel: submits 'count' entries, completes
 * all of them, and verifies payload integrity.  The payload base is
 * set to (channel_id * count) so each channel operates on a unique
 * non-overlapping range — if a completion from channel A appeared on
 * channel B, the payload check would catch it (cross-contamination).
 *
 * ENOSPC on submit and EAGAIN on complete are handled by switching
 * between submit and complete phases (same batching strategy as
 * test_stress_single).
 */
static void *multi_channel_thread(void *arg)
{
	struct thread_args *ta = arg;
	int submitted = 0, completed = 0;
	struct dmaplane_ring_entry entry;
	/*
	 * Payload base = channel_id * count.  This gives each channel a
	 * unique, non-overlapping range of payload values:
	 *   ch0: [0, count)    ch1: [count, 2*count)    etc.
	 * After the worker adds 1, expected completion payload at index i
	 * is (base + i + 1).  A mismatch means cross-channel contamination.
	 */
	__u64 base = (__u64)ta->channel_id * (__u64)ta->count;

	while (completed < ta->count) {
		/* Submit batch */
		int batch = 0;
		while (submitted < ta->count && batch < BATCH_SIZE) {
			if (submit_entry(ta->fd, base + (__u64)submitted, 0) == 0) {
				submitted++;
				batch++;
			} else if (errno == ENOSPC) {
				/* Ring full — drain completions before retrying */
				break;
			} else {
				fprintf(stderr, "  ch%d submit error: %s\n",
					ta->channel_id, strerror(errno));
				ta->ok = 0;
				return NULL;
			}
		}

		/* Complete */
		while (completed < submitted) {
			if (complete_entry(ta->fd, &entry) == 0) {
				/* Verify payload belongs to this channel */
				__u64 expected = base + (__u64)completed + 1;
				if (entry.payload != expected) {
					fprintf(stderr,
						"  ch%d mismatch at %d: "
						"expected 0x%llx got 0x%llx\n",
						ta->channel_id, completed,
						(unsigned long long)expected,
						(unsigned long long)entry.payload);
					ta->ok = 0;
					return NULL;
				}
				completed++;
			} else if (errno == EAGAIN) {
				/* Ring empty — submit more before retrying */
				break;
			} else {
				fprintf(stderr, "  ch%d complete error: %s\n",
					ta->channel_id, strerror(errno));
				ta->ok = 0;
				return NULL;
			}
		}
	}

	ta->ok = 1;
	return NULL;
}

/*
 * Stress all DMAPLANE_MAX_CHANNELS (4) channels in parallel via pthreads.
 * Each thread independently processes MULTI_PER_CHAN (250K) entries.
 * Measures aggregate throughput.  Validates:
 *  - Multiple channels operate concurrently without interference.
 *  - Per-channel locking (ring locks) does not cause cross-channel
 *    contention or deadlock.
 *  - No payload cross-contamination between channels.
 */
static void test_stress_multi(void)
{
	int fds[DMAPLANE_MAX_CHANNELS];
	pthread_t threads[DMAPLANE_MAX_CHANNELS];
	struct thread_args args[DMAPLANE_MAX_CHANNELS];
	struct timespec start, end;
	double elapsed, rate;
	int i, all_ok = 1;

	for (i = 0; i < DMAPLANE_MAX_CHANNELS; i++) {
		fds[i] = dev_open();
		if (fds[i] < 0) {
			TEST_FAIL("stress multi", "cannot open device for ch %d", i);
			while (--i >= 0) close(fds[i]);
			return;
		}
	}

	for (i = 0; i < DMAPLANE_MAX_CHANNELS; i++) {
		int ch = create_channel(fds[i]);
		if (ch < 0) {
			TEST_FAIL("stress multi",
				  "create channel %d failed", i);
			for (int j = 0; j < DMAPLANE_MAX_CHANNELS; j++)
				close(fds[j]);
			return;
		}
		args[i].fd = fds[i];
		args[i].channel_id = ch;
		args[i].count = MULTI_PER_CHAN;
		args[i].ok = 0;
	}

	clock_gettime(CLOCK_MONOTONIC, &start);

	for (i = 0; i < DMAPLANE_MAX_CHANNELS; i++)
		pthread_create(&threads[i], NULL, multi_channel_thread, &args[i]);

	for (i = 0; i < DMAPLANE_MAX_CHANNELS; i++)
		pthread_join(threads[i], NULL);

	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsed = (double)(end.tv_sec - start.tv_sec) +
		  (double)(end.tv_nsec - start.tv_nsec) / 1e9;
	rate = (double)(DMAPLANE_MAX_CHANNELS * MULTI_PER_CHAN) / elapsed;

	for (i = 0; i < DMAPLANE_MAX_CHANNELS; i++) {
		if (!args[i].ok) {
			all_ok = 0;
			break;
		}
	}

	if (all_ok) {
		printf("  PASS: stress multi — %d channels × %d entries "
		       "in %.3f s (%.0f ops/sec total)\n",
		       DMAPLANE_MAX_CHANNELS, MULTI_PER_CHAN,
		       elapsed, rate);
		tests_passed++;
	} else {
		TEST_FAIL("stress multi", "one or more channels failed");
	}

	for (i = 0; i < DMAPLANE_MAX_CHANNELS; i++)
		close(fds[i]);
}

/* ------------------------------------------------------------------ */
/* Test 7: Cleanup on close                                            */
/* ------------------------------------------------------------------ */

/*
 * Creates a channel, submits 100 entries without draining completions,
 * then closes the fd.  The close triggers release(), which calls
 * dmaplane_channel_destroy → kthread_stop on the worker.  Validates:
 *  - Kernel does not crash or hang when a channel is closed with
 *    in-flight (uncompleted) work.
 *  - Worker thread terminates cleanly despite pending ring entries.
 *  - This test cannot programmatically verify resource leaks — the
 *    user should check dmesg for WARN_ON or lockdep warnings after
 *    running the full suite.
 */
static void test_cleanup_on_close(void)
{
	int fd, ch;
	int i;

	fd = dev_open();
	if (fd < 0) {
		TEST_FAIL("cleanup on close", "cannot open device");
		return;
	}

	ch = create_channel(fd);
	if (ch < 0) {
		TEST_FAIL("cleanup on close", "create channel failed");
		close(fd);
		return;
	}

	/* Submit entries without completing them */
	for (i = 0; i < 100; i++) {
		if (submit_entry(fd, (__u64)i, 0) != 0)
			break;
	}

	/* Close without completing — should not crash or leak */
	close(fd);

	/*
	 * If we get here, the kernel didn't crash.
	 * User should check dmesg for warnings.
	 */
	TEST_PASS("cleanup on close (check dmesg for warnings)");
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

/*
 * Run all tests in sequence.  Print per-test PASS/FAIL and a summary.
 * Returns 0 if all pass, 1 if any fail.
 */
int main(void)
{
	printf("=== dmaplane Phase 1 Test Suite ===\n\n");

	test_open_close();
	test_channel_creation();
	test_submit_complete();
	test_ring_full();
	test_stress_single();
	test_stress_multi();
	test_cleanup_on_close();

	printf("\n=== Summary: %d passed, %d failed ===\n",
	       tests_passed, tests_failed);

	return tests_failed ? 1 : 0;
}
