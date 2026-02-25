// SPDX-License-Identifier: GPL-2.0
/*
 * dmabuf_rdma.c — Buffer management for DMA-mapped memory
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * This file allocates physical pages that will later be registered
 * with the RDMA NIC as Memory Regions (MRs) in Phase 4.  Kernel page
 * ownership is the foundation of the zero-copy architecture — the module
 * allocates, owns, and controls the lifecycle of all physical pages.
 *
 * Two allocation paths:
 *   BUF_TYPE_PAGES (preferred for RDMA): individual order-0 pages
 *     vmapped into a contiguous kernel VA.  These pages will be handed
 *     to ib_dma_map_sg() during MR registration (Phase 4).
 *   BUF_TYPE_COHERENT: dma_alloc_coherent for physically contiguous
 *     memory.  Useful for small control structures.  Cannot be used for
 *     MR registration (no page array for SG table construction).
 *
 * Both paths support mmap to userspace, completing the zero-copy
 * chain: userspace writes → same physical pages → NIC reads for TX.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>
#include <linux/numa.h>
#include <linux/nodemask.h>

#include "dmabuf_rdma.h"
#include "dmaplane_trace.h"

/*
 * dmabuf_rdma_create_buffer — Allocate a DMA buffer with NUMA awareness.
 *
 * Validates size, alloc_type, and NUMA node, finds a free slot in the
 * buffer array, allocates memory via the requested path, and returns
 * the handle in params->buf_id.
 *
 * NUMA placement (Phase 5):
 *   DMAPLANE_NUMA_ANY (-1): allocate on the current CPU's local node
 *     (same as plain alloc_page behavior).
 *   0..N: request allocation on that node via alloc_pages_node().  This
 *     is best-effort — the buddy allocator tries the target node's
 *     zonelist first, but falls back SILENTLY to distant nodes under
 *     memory pressure.  Post-hoc verification via page_to_nid() is the
 *     only way to detect misplacement.
 *
 * Concurrency: acquires buf_lock for the duration of slot search,
 * allocation, and bookkeeping.  The lock is held during allocation
 * because we need the slot to remain ours — no other thread should
 * claim it between the free-slot scan and the in_use=true write.
 */
int dmabuf_rdma_create_buffer(struct dmaplane_dev *dev,
			      struct dmaplane_buf_params *params)
{
	struct dmaplane_buffer *buf;
	unsigned int nr_pages;
	unsigned int i;
	int slot;
	int target_node;
	int local_count = 0, remote_count = 0;
	ktime_t alloc_start = ktime_get();

	/* Validate inputs */
	if (params->size == 0 || params->size > DMAPLANE_MAX_BUF_SIZE)
		return -EINVAL;

	if (params->alloc_type != DMAPLANE_BUF_TYPE_COHERENT &&
	    params->alloc_type != DMAPLANE_BUF_TYPE_PAGES)
		return -EINVAL;

	/*
	 * NUMA node validation — before acquiring the mutex so we
	 * fail fast on invalid input without blocking other callers.
	 */
	target_node = params->numa_node;
	if (target_node != DMAPLANE_NUMA_ANY) {
		if (target_node < 0 || target_node >= MAX_NUMNODES ||
		    !node_online(target_node)) {
			pr_err("invalid NUMA node %d\n", target_node);
			return -EINVAL;
		}
	}

	mutex_lock(&dev->buf_lock);

	/* Linear scan for free slot — deliberate simplicity for 64 slots */
	slot = -1;
	for (i = 0; i < DMAPLANE_MAX_BUFFERS; i++) {
		if (!dev->buffers[i].in_use) {
			slot = i;
			break;
		}
	}
	if (slot < 0) {
		mutex_unlock(&dev->buf_lock);
		return -ENOMEM;
	}

	buf = &dev->buffers[slot];

	/* Zero the slot to clear stale pointers from a previous occupant */
	memset(buf, 0, sizeof(*buf));

	/* Assign monotonically increasing handle, skip 0 on wrap */
	buf->id = dev->next_buf_id++;
	if (dev->next_buf_id == 0)
		dev->next_buf_id = 1;	/* 0 is reserved as "no buffer" sentinel */

	buf->alloc_type = params->alloc_type;
	buf->size = params->size;
	buf->numa_node = target_node;

	switch (params->alloc_type) {
	case DMAPLANE_BUF_TYPE_COHERENT:
		/*
		 * dma_alloc_coherent returns physically contiguous, cache-coherent
		 * memory with a single DMA address.  The kernel and device see
		 * consistent data without explicit cache flushes.
		 *
		 * dma_alloc_coherent does NOT accept a NUMA node parameter —
		 * it allocates from the device's local node (determined by the
		 * platform_device's NUMA affinity).  actual_numa_node is set
		 * post-hoc via page_to_nid() for informational reporting only.
		 *
		 * Uses &dev->pdev->dev — the platform device, not the char device.
		 */
		buf->vaddr = dma_alloc_coherent(&dev->pdev->dev, buf->size,
						&buf->dma_handle, GFP_KERNEL);
		if (!buf->vaddr) {
			mutex_unlock(&dev->buf_lock);
			return -ENOMEM;
		}
		/*
		 * Zero the memory — prevents leaking kernel data to userspace
		 * via mmap.  dma_alloc_coherent does not guarantee zeroing.
		 */
		memset(buf->vaddr, 0, buf->size);

		/* Best-effort: report node of the underlying pages.
		 * page_to_nid(virt_to_page()) works on x86-64 but is
		 * architecture-dependent — used for reporting only. */
		buf->actual_numa_node = page_to_nid(virt_to_page(buf->vaddr));
		if (target_node == DMAPLANE_NUMA_ANY)
			atomic64_inc(&dev->stats.numa_anon_allocs);
		else if (buf->actual_numa_node == target_node)
			atomic64_inc(&dev->stats.numa_local_allocs);
		else
			atomic64_inc(&dev->stats.numa_remote_allocs);
		break;

	case DMAPLANE_BUF_TYPE_PAGES:
		/*
		 * NUMA-aware page allocation.
		 *
		 * alloc_pages_node(nid, gfp, order) selects which NUMA node's
		 * zonelist to walk first.  Each NUMA node maintains independent
		 * free page pools per zone (ZONE_DMA, ZONE_NORMAL, etc.).
		 *
		 * If the target node is exhausted, the allocator falls back
		 * SILENTLY to progressively more distant nodes in the zonelist
		 * (ordered by ACPI SLIT distance).  There is no strict/fail
		 * mode — alloc_pages_node is always best-effort.
		 *
		 * For DMAPLANE_NUMA_ANY we pass NUMA_NO_NODE, which skips
		 * node targeting and uses the current CPU's local node —
		 * equivalent to plain alloc_pages(GFP_KERNEL, 0).
		 *
		 * __GFP_ZERO: critical — prevents leaking kernel memory to
		 * userspace via mmap.
		 */
		nr_pages = DIV_ROUND_UP(buf->size, PAGE_SIZE);
		buf->pages = kvcalloc(nr_pages, sizeof(struct page *), GFP_KERNEL);
		if (!buf->pages) {
			mutex_unlock(&dev->buf_lock);
			return -ENOMEM;
		}

		for (i = 0; i < nr_pages; i++) {
			/* NUMA_NO_NODE → allocator uses current CPU's local
			 * node; specific nid → tries that node's buddy lists
			 * first, falls back to distant nodes under pressure. */
			int alloc_node = (target_node == DMAPLANE_NUMA_ANY)
					  ? NUMA_NO_NODE : target_node;

			buf->pages[i] = alloc_pages_node(alloc_node,
							 GFP_KERNEL | __GFP_ZERO, 0);
			if (!buf->pages[i]) {
				/* Unwind: free all pages allocated so far */
				while (i > 0)
					__free_page(buf->pages[--i]);
				kvfree(buf->pages);
				buf->pages = NULL;
				mutex_unlock(&dev->buf_lock);
				return -ENOMEM;
			}

			/* Post-hoc placement verification.
			 * page_to_nid() reads the page's node from its
			 * struct page flags — the only way to detect
			 * silent fallback by the buddy allocator. */
			if (target_node != DMAPLANE_NUMA_ANY) {
				int actual = page_to_nid(buf->pages[i]);

				if (actual == target_node)
					local_count++;
				else
					remote_count++;
			}
		}
		buf->nr_pages = nr_pages;

		/* Determine actual NUMA node (majority vote across pages) */
		if (target_node == DMAPLANE_NUMA_ANY) {
			buf->actual_numa_node = (nr_pages > 0)
				? page_to_nid(buf->pages[0]) : -1;
			atomic64_inc(&dev->stats.numa_anon_allocs);
		} else {
			buf->actual_numa_node = (local_count >= remote_count)
				? target_node
				: page_to_nid(buf->pages[0]);
			if (remote_count == 0)
				atomic64_inc(&dev->stats.numa_local_allocs);
			else
				atomic64_inc(&dev->stats.numa_remote_allocs);
		}

		if (remote_count > 0)
			pr_warn("buffer NUMA placement: %d/%u pages on node %d, %d misplaced\n",
				local_count, nr_pages, target_node, remote_count);

		/*
		 * vmap: create a contiguous kernel virtual address from scattered
		 * physical pages.  The vmapped address is used by the RDMA engine
		 * (Phase 4) as the SGE virtual address for rxe's memcpy-based
		 * send path.
		 */
		buf->vaddr = vmap(buf->pages, nr_pages, VM_MAP, PAGE_KERNEL);
		if (!buf->vaddr) {
			for (i = 0; i < nr_pages; i++)
				__free_page(buf->pages[i]);
			kvfree(buf->pages);
			buf->pages = NULL;
			mutex_unlock(&dev->buf_lock);
			return -ENOMEM;
		}
		break;

	default:
		/* Should not reach here — validated above */
		mutex_unlock(&dev->buf_lock);
		return -EINVAL;
	}

	buf->in_use = true;
	atomic_set(&buf->mmap_count, 0);

	/* Fill output fields for userspace */
	params->buf_id = buf->id;
	params->actual_numa_node = buf->actual_numa_node;

	atomic64_inc(&dev->stats.buffers_created);

	trace_dmaplane_buf_alloc(buf->id, params->size, params->alloc_type,
				 buf->actual_numa_node,
				 ktime_to_ns(ktime_sub(ktime_get(),
						       alloc_start)));

	pr_debug("buffer %u created: type=%d size=%zu nr_pages=%u numa_req=%d numa_actual=%d\n",
		 buf->id, buf->alloc_type, buf->size, buf->nr_pages,
		 buf->numa_node, buf->actual_numa_node);

	mutex_unlock(&dev->buf_lock);
	return 0;
}
EXPORT_SYMBOL_GPL(dmabuf_rdma_create_buffer);

/*
 * dmabuf_rdma_destroy_buffer — Free a buffer and its backing memory.
 *
 * Refuses destruction if userspace has active mmap references (mmap_count > 0)
 * to prevent use-after-free.  Cleanup is reverse-order of allocation.
 *
 * Concurrency: acquires buf_lock.  The lock protects the buffer lookup
 * and the in_use flag transition.  The actual memory freeing happens
 * under the lock because we need to prevent a concurrent mmap from
 * finding the buffer while its pages are being freed.
 */
int dmabuf_rdma_destroy_buffer(struct dmaplane_dev *dev, unsigned int buf_id)
{
	struct dmaplane_buffer *buf;
	unsigned int i;

	mutex_lock(&dev->buf_lock);

	buf = dmabuf_rdma_find_buffer(dev, buf_id);
	if (!buf) {
		mutex_unlock(&dev->buf_lock);
		return -ENOENT;
	}

	/* Refuse if userspace still has active mappings */
	if (atomic_read(&buf->mmap_count) > 0) {
		pr_debug("buffer %u: cannot destroy — %d active mmaps\n",
			 buf->id, atomic_read(&buf->mmap_count));
		mutex_unlock(&dev->buf_lock);
		return -EBUSY;
	}

	/* Refuse if buffer is exported as dma-buf — importers may still
	 * hold references.  The release callback clears this flag when
	 * the last dma-buf reference drops. */
	if (buf->dmabuf_exported) {
		pr_debug("buffer %u: cannot destroy — dma-buf exported\n",
			 buf->id);
		mutex_unlock(&dev->buf_lock);
		return -EBUSY;
	}

	/* Reverse-order cleanup based on allocation type */
	switch (buf->alloc_type) {
	case DMAPLANE_BUF_TYPE_PAGES:
		if (buf->vaddr)
			vunmap(buf->vaddr);
		if (buf->pages) {
			for (i = 0; i < buf->nr_pages; i++)
				__free_page(buf->pages[i]);
			kvfree(buf->pages);
		}
		break;

	case DMAPLANE_BUF_TYPE_COHERENT:
		if (buf->vaddr)
			dma_free_coherent(&dev->pdev->dev, buf->size,
					  buf->vaddr, buf->dma_handle);
		break;
	}

	pr_debug("buffer %u destroyed\n", buf->id);

	buf->in_use = false;
	atomic64_inc(&dev->stats.buffers_destroyed);

	mutex_unlock(&dev->buf_lock);
	return 0;
}
EXPORT_SYMBOL_GPL(dmabuf_rdma_destroy_buffer);

/*
 * dmabuf_rdma_find_buffer — Look up a buffer by ID.
 *
 * Caller MUST hold buf_lock for the lifetime of the returned pointer.
 * The buffer could be destroyed by another thread between find and use
 * if the lock is not held.
 */
struct dmaplane_buffer *dmabuf_rdma_find_buffer(struct dmaplane_dev *dev,
						unsigned int buf_id)
{
	unsigned int i;

	for (i = 0; i < DMAPLANE_MAX_BUFFERS; i++) {
		if (dev->buffers[i].in_use && dev->buffers[i].id == buf_id)
			return &dev->buffers[i];
	}
	return NULL;
}
EXPORT_SYMBOL_GPL(dmabuf_rdma_find_buffer);

/*
 * dmabuf_rdma_find_mr - Linear scan for an MR by ID.
 *
 * Caller MUST hold mr_lock for the lifetime of the returned pointer.
 * The MR could be deregistered by another thread between find and use
 * if the lock is not held.
 */
struct dmaplane_mr_entry *dmabuf_rdma_find_mr(struct dmaplane_dev *dev,
					      __u32 mr_id)
{
	unsigned int i;

	for (i = 0; i < DMAPLANE_MAX_MRS; i++) {
		if (dev->mrs[i].in_use && dev->mrs[i].id == mr_id)
			return &dev->mrs[i];
	}
	return NULL;
}
EXPORT_SYMBOL_GPL(dmabuf_rdma_find_mr);
