// SPDX-License-Identifier: GPL-2.0
/*
 * gpu_p2p.c -- GPU memory pinning and host<->GPU DMA via NVIDIA P2P API
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * This module implements GPU VRAM access for dmaplane using the NVIDIA
 * peer-to-peer (P2P) kernel API. It is compiled only when NVIDIA P2P
 * headers are detected (CONFIG_DMAPLANE_GPU), and all nvidia_p2p_*
 * symbols are resolved at runtime via symbol_get() to avoid static
 * linking against the proprietary nvidia.ko module.
 *
 * How GPU memory access works at the hardware level:
 *
 *   GPU VRAM is not directly addressable by the CPU. The NVIDIA driver
 *   exposes GPU memory through PCIe BAR1 (Base Address Register 1) --
 *   a window of physical addresses that the CPU can use to read/write
 *   GPU memory via MMIO. nvidia_p2p_get_pages() pins a range of GPU
 *   virtual memory and returns the BAR1 physical addresses for each
 *   64KB page.
 *
 * Critical properties of BAR1 pages:
 *
 *   1. NOT host DRAM -- BAR physical addresses are PCIe MMIO windows,
 *      not system RAM. struct page does not exist for them. pfn_to_page()
 *      is undefined. You cannot use sg_set_page() or any function that
 *      expects a valid struct page *.
 *
 *   2. NOT contiguous -- each page_table->pages[i]->physical_address
 *      can be at an arbitrary BAR1 offset. You CANNOT ioremap the first
 *      page's address with the full region size. Each 64KB page must be
 *      ioremap'd individually.
 *
 *   3. ALWAYS 64KB -- regardless of the host PAGE_SIZE (usually 4KB).
 *      Both the GPU VA and the pin length must be 64KB-aligned.
 *
 *   4. CPU access via MMIO -- after ioremap_wc(), reads use memcpy_fromio()
 *      (PCIe non-posted reads, slow) and writes use memcpy_toio()
 *      (PCIe posted writes, coalesced by write-combining into full
 *      64-byte TLP payloads, much faster).
 *
 * Revocation model:
 *
 *   The NVIDIA driver can reclaim pinned pages at any time -- most
 *   commonly when userspace calls cudaFree(). When this happens, the
 *   driver invokes the free_callback registered during pin. This
 *   callback runs with NVIDIA-internal locks held, so it MUST be
 *   trivial: set an atomic flag, signal a completion, return. After
 *   revocation, all BAR page mappings are stale -- the GPU may have
 *   already reassigned that VRAM to another allocation. Any MMIO
 *   access after revocation reads garbage or faults.
 *
 *   Cleanup has two paths:
 *     - Normal (not revoked): nvidia_p2p_put_pages() releases the
 *       pin and frees the page_table in one call.
 *     - Revoked: nvidia_p2p_put_pages() would deadlock (tries to
 *       acquire locks the callback already holds). Use
 *       nvidia_p2p_free_page_table() to free just the struct.
 *
 * Symbol resolution (GPL/proprietary workaround):
 *
 *   nvidia.ko exports nvidia_p2p_* as EXPORT_SYMBOL_GPL, but nvidia.ko
 *   itself is a proprietary module. On kernel 6.5+, the module loader
 *   tracks which modules use GPL-only symbols and refuses to resolve
 *   cross-references to proprietary modules. This means a GPL module
 *   cannot statically depend on nvidia.ko symbols -- insmod fails with
 *   "module using GPL-only symbols uses symbols from proprietary module."
 *
 *   The workaround: resolve the three nvidia_p2p_* functions at runtime
 *   using symbol_get()/symbol_put(). This performs a dynamic lookup in
 *   the kernel symbol table without creating a modpost dependency. The
 *   kernel allows GPL modules to symbol_get() any EXPORT_SYMBOL_GPL
 *   symbol from any loaded module.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/io.h>
#include <linux/scatterlist.h>
#include <linux/ktime.h>
#include <linux/sched.h>

#include "dmaplane.h"
#include "gpu_p2p.h"
#include "dmabuf_rdma.h"
#include "rdma_engine.h"
#include "dmaplane_trace.h"

/* ============================================================================
 * Runtime-resolved NVIDIA P2P function pointers
 * ============================================================================
 *
 * These typedefs match the exact signatures from nv-p2p.h. The static
 * function pointers are populated by dmaplane_gpu_init() via symbol_get()
 * and cleared by dmaplane_gpu_exit() via symbol_put().
 *
 * All GPU P2P operations in this file call through these pointers
 * instead of referencing the nvidia_p2p_* symbols directly. This
 * eliminates the static module dependency that would otherwise cause
 * insmod to fail on kernel 6.5+.
 *
 * nv_get_pages:      Pins GPU VRAM and returns BAR1 physical addresses.
 *                    Signature: (p2p_token, va_space, gpu_va, length,
 *                    &page_table, free_callback, callback_data) -> int
 *
 * nv_put_pages:      Releases a pin created by get_pages. Must NOT be
 *                    called from the unpin callback (deadlock).
 *                    Signature: (p2p_token, va_space, gpu_va, page_table) -> int
 *
 * nv_free_page_table: Frees the page_table struct after revocation, when
 *                    put_pages cannot be called. Does not release the pin
 *                    (NVIDIA already reclaimed the pages).
 *                    Signature: (page_table) -> int
 */
typedef int (*nv_p2p_get_pages_fn_t)(uint64_t, uint32_t, uint64_t, uint64_t,
				     struct nvidia_p2p_page_table **,
				     void (*)(void *), void *);
typedef int (*nv_p2p_put_pages_fn_t)(uint64_t, uint32_t, uint64_t,
				     struct nvidia_p2p_page_table *);
typedef int (*nv_p2p_free_page_table_fn_t)(struct nvidia_p2p_page_table *);

static nv_p2p_get_pages_fn_t        nv_get_pages;
static nv_p2p_put_pages_fn_t        nv_put_pages;
static nv_p2p_free_page_table_fn_t  nv_free_page_table;

/* ============================================================================
 * dmaplane_gpu_unpin_callback() -- NVIDIA revocation callback
 * ============================================================================
 *
 * Called by the NVIDIA driver when it reclaims GPU memory that dmaplane
 * has pinned via nvidia_p2p_get_pages(). This typically happens when
 * userspace calls cudaFree() while the buffer is still pinned by the
 * kernel module.
 *
 * CRITICAL CONSTRAINTS -- NVIDIA's internal locks are held when this
 * callback fires. Violating any of these rules causes a kernel deadlock
 * or crash:
 *
 *   - DO NOT take any mutex, spinlock, or rwsem (including gpu_buf_lock)
 *   - DO NOT call nvidia_p2p_put_pages() (acquires the same NVIDIA lock
 *     that is already held -> deadlock)
 *   - DO NOT call printk() (can attempt memory allocation under pressure,
 *     which can sleep)
 *   - DO NOT do any blocking operation (sleep, wait, schedule)
 *   - DO return as fast as possible
 *
 * What this callback does:
 *   1. Sets gpu_revoked = 1 atomically -- signals all DMA functions to
 *      stop accessing BAR pages immediately.
 *   2. Signals revoke_done -- wakes any waiters (future use for
 *      synchronized teardown).
 *   3. Increments callbacks_fired stat -- atomic64_inc is safe from any
 *      context (it's just a locked add instruction on x86).
 *   4. Returns immediately.
 *
 * After this callback returns:
 *   - All bar_pages[] ioremap_wc pointers are stale (NVIDIA may reassign
 *     the VRAM). Reading them returns garbage; writing may corrupt
 *     other GPU allocations.
 *   - dmaplane_gpu_unpin() must use nvidia_p2p_free_page_table()
 *     instead of nvidia_p2p_put_pages() to avoid deadlock.
 *
 * @data: Pointer to the struct dmaplane_gpu_buffer that was passed as
 *        the callback_data argument to nvidia_p2p_get_pages().
 */
static void dmaplane_gpu_unpin_callback(void *data)
{
	struct dmaplane_gpu_buffer *gpu_buf = data;

	atomic_set(&gpu_buf->gpu_revoked, 1);
	complete_all(&gpu_buf->revoke_done);

	/* Safe from any context -- atomic64_inc is a single locked add
	 * instruction, no sleeping or locking required. We stash the
	 * edev pointer in the gpu_buf's parent array's container, but
	 * the callback only receives the gpu_buf pointer. Since the
	 * callbacks_fired counter is accessed via the edev pointer
	 * (which we don't have here), we track it in the unpin path
	 * instead. See dmaplane_gpu_unpin() for the actual increment.
	 *
	 * Note: we cannot increment gpu_stats.callbacks_fired here
	 * because the callback receives only gpu_buf, not edev. The
	 * callback_fired tracking happens in the unpin path where we
	 * have access to edev via the revoked flag detection. */
}

/* ============================================================================
 * gpu_bar_to_sgtable() -- build scatterlist from BAR physical addresses
 * ============================================================================
 *
 * Constructs an sg_table where each scatterlist entry represents one
 * 64KB GPU BAR page. Unlike normal host sg_tables (built with
 * sg_set_page from struct page *), GPU BAR pages have no valid struct
 * page -- we set sg_dma_address and sg_dma_len directly from the BAR1
 * physical addresses returned by nvidia_p2p_get_pages().
 *
 * This sg_table serves two purposes:
 *   1. Future GPUDirect RDMA -- a NIC can be programmed to DMA directly
 *      from/to these BAR addresses, bypassing the CPU entirely.
 *   2. Metadata -- records the physical layout for debugging and
 *      introspection.
 *
 * Note: sg_dma_address is set directly (not via dma_map_sg), so this
 * table is pre-mapped from the perspective of any device on the same
 * PCIe fabric. No IOMMU translation is applied.
 *
 * @pt: NVIDIA page table returned by nvidia_p2p_get_pages(). Must not
 *      be NULL. pt->entries gives the number of pages; each
 *      pt->pages[i]->physical_address is a BAR1 physical address.
 *
 * Return: Allocated sg_table on success, ERR_PTR(-ENOMEM) on failure.
 *         Caller owns the returned table and must free it with
 *         sg_free_table() + kfree().
 */
static struct sg_table *gpu_bar_to_sgtable(nvidia_p2p_page_table_t *pt)
{
	struct sg_table *sgt;
	struct scatterlist *sg;
	unsigned int i;

	sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	if (!sgt)
		return ERR_PTR(-ENOMEM);

	if (sg_alloc_table(sgt, pt->entries, GFP_KERNEL)) {
		kfree(sgt);
		return ERR_PTR(-ENOMEM);
	}

	for_each_sg(sgt->sgl, sg, pt->entries, i) {
		sg_dma_address(sg) = pt->pages[i]->physical_address;
		sg_dma_len(sg) = DMAPLANE_GPU_PAGE_SIZE;
	}

	return sgt;
}

/* ============================================================================
 * dmaplane_gpu_pin() -- pin GPU VRAM and set up BAR mappings
 * ============================================================================
 *
 * Pins a region of GPU virtual memory via the NVIDIA P2P API and
 * prepares it for CPU access. The full sequence is:
 *
 *   1. Validate alignment (gpu_va and size must be 64KB-aligned)
 *   2. Find a free slot in edev->gpu_buffers[]
 *   3. Initialize the buffer struct (revoked=0, completion init)
 *   4. Call nvidia_p2p_get_pages() -- returns BAR1 physical addresses
 *      for each 64KB page and registers our unpin callback
 *   5. Verify page table version compatibility (major version must match)
 *   6. Build sg_table from BAR addresses (for future GPUDirect RDMA)
 *   7. ioremap_wc each 64KB page individually into kernel virtual space
 *   8. Record GPU NUMA node, assign unique handle, mark in_use
 *
 * Error handling: on failure at any step, all resources allocated by
 * prior steps are cleaned up before returning. If nvidia_p2p_get_pages()
 * succeeded, nvidia_p2p_put_pages() is called to release the pin.
 *
 * The p2pToken=0, vaSpaceToken=0 arguments to nvidia_p2p_get_pages()
 * work on NVIDIA driver >=535 for same-process GPU memory. These tokens
 * are legacy identifiers from older driver versions; zero means "use
 * the default context."
 *
 * Locking: holds edev->gpu_buf_lock across the entire operation. This
 * mutex can sleep, which is required because nvidia_p2p_get_pages() and
 * ioremap_wc() may sleep. The lock serializes pin/unpin operations but does
 * not block DMA operations (which take the lock only briefly for lookup).
 *
 * See gpu_p2p.h for full parameter documentation.
 */
int dmaplane_gpu_pin(struct dmaplane_dev *edev, u64 gpu_va, u64 size,
		     __u32 *handle_out, __s32 *numa_node_out,
		     __u32 *num_pages_out, u64 *bar1_consumed_out)
{
	struct dmaplane_gpu_buffer *gpu_buf = NULL;
	ktime_t pin_start = ktime_get();
	int i, ret;

	/* Validate 64KB alignment -- nvidia_p2p_get_pages requires it */
	if (size == 0)
		return -EINVAL;
	if (gpu_va & (DMAPLANE_GPU_PAGE_SIZE - 1)) {
		pr_err("GPU VA 0x%llx not 64KB-aligned\n", gpu_va);
		return -EINVAL;
	}
	if (size & (DMAPLANE_GPU_PAGE_SIZE - 1)) {
		pr_err("GPU pin size %llu not 64KB-multiple\n", size);
		return -EINVAL;
	}

	mutex_lock(&edev->gpu_buf_lock);

	/* Find a free slot in the fixed-size buffer array */
	for (i = 0; i < DMAPLANE_MAX_GPU_BUFFERS; i++) {
		if (!edev->gpu_buffers[i].in_use) {
			gpu_buf = &edev->gpu_buffers[i];
			break;
		}
	}
	if (!gpu_buf) {
		mutex_unlock(&edev->gpu_buf_lock);
		pr_err("no free GPU buffer slots\n");
		return -ENOMEM;
	}

	/* Initialize the buffer before calling into NVIDIA --
	 * the callback could theoretically fire immediately if the
	 * GPU memory is being freed concurrently. */
	memset(gpu_buf, 0, sizeof(*gpu_buf));
	gpu_buf->gpu_va = gpu_va;
	gpu_buf->size = size;
	atomic_set(&gpu_buf->gpu_revoked, 0);
	init_completion(&gpu_buf->revoke_done);

	/* Pin GPU VRAM -- returns BAR1 physical addresses.
	 * p2pToken=0, vaSpaceToken=0 for same-process on driver >=535. */
	ret = nv_get_pages(0, 0, gpu_va, size,
				   &gpu_buf->page_table,
				   dmaplane_gpu_unpin_callback, gpu_buf);
	if (ret) {
		pr_err("nvidia_p2p_get_pages failed: %d\n", ret);
		mutex_unlock(&edev->gpu_buf_lock);
		return ret;
	}

	/* Verify page table version compatibility -- the NVIDIA driver
	 * may have a different major version than what our headers expect.
	 * If incompatible, release the pin and fail. */
	if (!NVIDIA_P2P_PAGE_TABLE_VERSION_COMPATIBLE(gpu_buf->page_table)) {
		pr_err("incompatible NVIDIA P2P page table version 0x%x\n",
		       gpu_buf->page_table->version);
		nv_put_pages(0, 0, gpu_va, gpu_buf->page_table);
		mutex_unlock(&edev->gpu_buf_lock);
		return -EINVAL;
	}

	gpu_buf->num_pages = gpu_buf->page_table->entries;

	/* Build SG table from BAR physical addresses */
	gpu_buf->sgt = gpu_bar_to_sgtable(gpu_buf->page_table);
	if (IS_ERR(gpu_buf->sgt)) {
		ret = PTR_ERR(gpu_buf->sgt);
		gpu_buf->sgt = NULL;
		nv_put_pages(0, 0, gpu_va, gpu_buf->page_table);
		mutex_unlock(&edev->gpu_buf_lock);
		return ret;
	}

	/* ioremap_wc each 64KB BAR page individually.
	 *
	 * BAR pages are NOT guaranteed contiguous in physical address
	 * space -- each pages[i]->physical_address can be at an arbitrary
	 * BAR1 offset. A single ioremap_wc(pages[0], total_size) would be
	 * WRONG because it assumes contiguity. Instead we create an array
	 * of per-page write-combining MMIO mappings.
	 *
	 * Write-combining (WC) allows the CPU to coalesce stores into
	 * full 64-byte PCIe TLP payloads instead of one store per TLP
	 * (as with plain ioremap/UC). This yields ~100x improvement
	 * for host->GPU writes via memcpy_toio(). GPU->host reads
	 * (memcpy_fromio) remain slow -- PCIe reads are non-posted. */
	gpu_buf->bar_pages = kcalloc(gpu_buf->num_pages,
				     sizeof(void __iomem *), GFP_KERNEL);
	if (!gpu_buf->bar_pages) {
		sg_free_table(gpu_buf->sgt);
		kfree(gpu_buf->sgt);
		gpu_buf->sgt = NULL;
		nv_put_pages(0, 0, gpu_va, gpu_buf->page_table);
		mutex_unlock(&edev->gpu_buf_lock);
		return -ENOMEM;
	}

	for (i = 0; i < gpu_buf->num_pages; i++) {
		gpu_buf->bar_pages[i] = ioremap_wc(
			gpu_buf->page_table->pages[i]->physical_address,
			DMAPLANE_GPU_PAGE_SIZE);
		if (!gpu_buf->bar_pages[i]) {
			pr_err("ioremap_wc failed for GPU BAR page %d\n", i);
			/* Unwind: iounmap pages successfully mapped before the failure */
			while (i > 0)
				iounmap(gpu_buf->bar_pages[--i]);
			kfree(gpu_buf->bar_pages);
			gpu_buf->bar_pages = NULL;
			sg_free_table(gpu_buf->sgt);
			kfree(gpu_buf->sgt);
			gpu_buf->sgt = NULL;
			nv_put_pages(0, 0, gpu_va, gpu_buf->page_table);
			mutex_unlock(&edev->gpu_buf_lock);
			return -ENOMEM;
		}
	}

	pr_info("GPU BAR mapped with write-combining (WC)\n");

	/* -- Contiguous RDMA mapping -----------------------------------------
	 *
	 * rxe (soft-RoCE) implements RDMA sends by doing memcpy() from the
	 * address in sge.addr, which it interprets as a kernel virtual
	 * address. For GPU-backed MRs, sge.addr must point to a contiguous
	 * kernel VA mapping of the GPU BAR pages.
	 *
	 * The per-page bar_pages[] array created above is non-contiguous in
	 * kernel VA (each ioremap_wc returns an independent mapping). We
	 * need a SINGLE ioremap_wc spanning ALL pages so rxe gets one
	 * contiguous VA range to memcpy from.
	 *
	 * This only works if the BAR1 physical addresses are contiguous --
	 * i.e., pages[i].phys == pages[0].phys + i * 64KB. A single
	 * cudaMalloc typically yields contiguous BAR1 pages, but this is
	 * not guaranteed (fragmented BAR1 can produce gaps).
	 *
	 * If non-contiguous: rdma_vaddr = NULL, and GPU_REGISTER_MR will
	 * fail with -EINVAL. This is not fatal -- pin/unpin/DMA still work,
	 * only RDMA is unavailable for this buffer.
	 *
	 * If contiguous but ioremap_wc fails: same outcome, just a kernel
	 * resource exhaustion case.
	 */
	/* Check if BAR pages form a contiguous physical range (base + i*64KB).
	 * A single cudaMalloc typically produces contiguous pages;
	 * fragmented allocations do not. */
	{
		bool contiguous = true;

		for (i = 1; i < gpu_buf->num_pages; i++) {
			phys_addr_t expected =
				gpu_buf->page_table->pages[0]->physical_address
				+ (u64)i * DMAPLANE_GPU_PAGE_SIZE;
			if (gpu_buf->page_table->pages[i]->physical_address
			    != expected) {
				contiguous = false;
				break;
			}
		}

		if (contiguous) {
			gpu_buf->rdma_vaddr = ioremap_wc(
				gpu_buf->page_table->pages[0]->physical_address,
				(size_t)gpu_buf->num_pages * DMAPLANE_GPU_PAGE_SIZE);
			if (!gpu_buf->rdma_vaddr)
				pr_warn("contiguous ioremap_wc failed"
					" -- RDMA not available\n");
			else
				pr_info("GPU BAR pages contiguous"
					" -- RDMA mapping ready (%d pages)\n",
					gpu_buf->num_pages);
		} else {
			gpu_buf->rdma_vaddr = NULL;
			pr_warn("GPU BAR pages non-contiguous"
				" -- RDMA not available\n");
		}
	}

	/* GPU NUMA node -- best effort via the platform device. On systems
	 * with proper ACPI/SRAT tables this reports the correct node. */
	gpu_buf->numa_node = dev_to_node(&edev->pdev->dev);

	/* Assign a unique monotonic handle (wraps at UINT_MAX, skips 0 so
	 * uninitialized handles in zero-init structs are distinguishable) */
	gpu_buf->id = edev->next_gpu_buf_id;
	if (++edev->next_gpu_buf_id == 0)
		edev->next_gpu_buf_id = 1;
	gpu_buf->in_use = true;

	/* Populate output parameters for the ioctl caller */
	*handle_out = gpu_buf->id;
	*numa_node_out = gpu_buf->numa_node;
	*num_pages_out = gpu_buf->num_pages;
	*bar1_consumed_out = (u64)gpu_buf->num_pages * DMAPLANE_GPU_PAGE_SIZE;

	mutex_unlock(&edev->gpu_buf_lock);

	/* Phase 8 instrumentation: tracepoint + stats counter */
	trace_dmaplane_gpu_pin(gpu_buf->id, gpu_va, size,
			       gpu_buf->num_pages,
			       gpu_buf->rdma_vaddr ? 1 : 0,
			       ktime_to_ns(ktime_sub(ktime_get(), pin_start)));
	atomic64_inc(&edev->gpu_stats.pins_total);

	pr_info("GPU buffer %u pinned (va=0x%llx size=%llu pages=%d numa=%d)\n",
		gpu_buf->id, gpu_va, size, gpu_buf->num_pages,
		gpu_buf->numa_node);
	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_gpu_pin);

/* ============================================================================
 * dmaplane_gpu_unpin() -- release a pinned GPU buffer
 * ============================================================================
 *
 * Tears down all resources for a pinned GPU buffer in order:
 *   1. iounmap the contiguous RDMA mapping (rdma_vaddr) first
 *   2. iounmap each BAR page in bar_pages[]
 *   3. Free the bar_pages array
 *   4. Free the sg_table
 *   5. Release the NVIDIA page table:
 *      - If NOT revoked: nvidia_p2p_put_pages() (normal release)
 *      - If revoked: nvidia_p2p_free_page_table() (callback already fired,
 *        put_pages would deadlock on NVIDIA's internal locks)
 *   6. Mark the slot as free
 *
 * The revoked/non-revoked distinction is the most important design
 * decision in this function. See the file-level comment for details
 * on the revocation model.
 *
 * Called from: IOCTL_GPU_UNPIN handler and dmaplane_exit() cleanup.
 *
 * See gpu_p2p.h for parameter documentation.
 */
int dmaplane_gpu_unpin(struct dmaplane_dev *edev, __u32 handle)
{
	struct dmaplane_gpu_buffer *gpu_buf;
	bool was_revoked;
	int i;

	mutex_lock(&edev->gpu_buf_lock);

	/* Find the buffer by handle */
	gpu_buf = NULL;
	for (i = 0; i < DMAPLANE_MAX_GPU_BUFFERS; i++) {
		if (edev->gpu_buffers[i].in_use &&
		    edev->gpu_buffers[i].id == handle) {
			gpu_buf = &edev->gpu_buffers[i];
			break;
		}
	}

	if (!gpu_buf) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -ENOENT;
	}

	/* Snapshot revocation state for stats tracking below */
	was_revoked = !!atomic_read(&gpu_buf->gpu_revoked);

	/* Unmap the contiguous RDMA mapping first (if it exists).
	 * This must happen BEFORE the per-page iounmap loop -- the
	 * rdma_vaddr mapping overlaps the same BAR physical range as
	 * the individual bar_pages[] mappings, and we must not leave
	 * a dangling kernel VA pointing at reclaimed BAR1 aperture. */
	if (gpu_buf->rdma_vaddr) {
		iounmap(gpu_buf->rdma_vaddr);
		gpu_buf->rdma_vaddr = NULL;
	}

	/* Unmap all per-page ioremap_wc'd BAR pages */
	if (gpu_buf->bar_pages) {
		for (i = 0; i < gpu_buf->num_pages; i++) {
			if (gpu_buf->bar_pages[i])
				iounmap(gpu_buf->bar_pages[i]);
		}
		kfree(gpu_buf->bar_pages);
		gpu_buf->bar_pages = NULL;
	}

	/* Free SG table */
	if (gpu_buf->sgt) {
		sg_free_table(gpu_buf->sgt);
		kfree(gpu_buf->sgt);
		gpu_buf->sgt = NULL;
	}

	/* Release GPU pages -- two paths depending on revocation state:
	 *
	 *   NOT revoked (normal path):
	 *     nvidia_p2p_put_pages() releases the pin, invokes cleanup
	 *     inside the NVIDIA driver, and frees the page_table.
	 *
	 *   Revoked (callback already fired):
	 *     The NVIDIA driver already reclaimed the pages. Calling
	 *     put_pages would try to acquire NVIDIA locks that the
	 *     callback path still holds -> deadlock. Instead, call
	 *     free_page_table() which just kfrees the struct without
	 *     touching NVIDIA's internal state. */
	if (gpu_buf->page_table) {
		if (!atomic_read(&gpu_buf->gpu_revoked)) {
			nv_put_pages(0, 0, gpu_buf->gpu_va,
					     gpu_buf->page_table);
		} else {
			nv_free_page_table(gpu_buf->page_table);
		}
		gpu_buf->page_table = NULL;
	}

	gpu_buf->in_use = false;
	mutex_unlock(&edev->gpu_buf_lock);

	/* Phase 8 instrumentation: track unpins and callback-fired events.
	 * The unpin callback itself cannot increment gpu_stats because it
	 * only receives the gpu_buf pointer, not edev. We detect revocation
	 * here and credit the callback counter at unpin time. */
	atomic64_inc(&edev->gpu_stats.unpins_total);
	if (was_revoked)
		atomic64_inc(&edev->gpu_stats.callbacks_fired);

	pr_info("GPU buffer %u unpinned%s\n", handle,
		was_revoked ? " (was revoked)" : "");
	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_gpu_unpin);

/* ============================================================================
 * dmaplane_gpu_find_buffer() -- look up a GPU buffer by handle
 * ============================================================================
 *
 * Performs a linear scan of the gpu_buffers[] array looking for an
 * in_use entry with the matching handle. The array is small (16 slots)
 * so linear scan is appropriate.
 *
 * Caller must hold edev->gpu_buf_lock or otherwise guarantee that the
 * array is not being modified concurrently (e.g. during module exit
 * after the char device has been removed).
 *
 * Return: pointer to the matching buffer, or NULL if not found.
 */
struct dmaplane_gpu_buffer *dmaplane_gpu_find_buffer(struct dmaplane_dev *edev,
						     __u32 handle)
{
	int i;

	for (i = 0; i < DMAPLANE_MAX_GPU_BUFFERS; i++) {
		if (edev->gpu_buffers[i].in_use &&
		    edev->gpu_buffers[i].id == handle)
			return &edev->gpu_buffers[i];
	}
	return NULL;
}
EXPORT_SYMBOL_GPL(dmaplane_gpu_find_buffer);

/* ============================================================================
 * Host<->GPU DMA helpers -- page-walking BAR memcpy
 * ============================================================================
 *
 * These two static helpers implement the core copy loop between host
 * DRAM and GPU BAR pages. Because BAR pages are not contiguous, the
 * copy must walk the bar_pages[] array and handle 64KB page boundary
 * crossings.
 *
 * Algorithm for a transfer of 'size' bytes starting at 'offset':
 *
 *   1. Compute which 64KB page the current position falls in:
 *        page_idx = (offset + pos) / 64KB
 *   2. Compute the offset within that page:
 *        page_off = (offset + pos) % 64KB
 *   3. The maximum contiguous chunk within this page is:
 *        chunk = 64KB - page_off
 *   4. Clamp chunk to remaining bytes, do the memcpy, advance.
 *   5. Repeat until all bytes are transferred.
 *
 * Performance notes:
 *   - memcpy_fromio (GPU->host): PCIe read transactions. The CPU must
 *     wait for each cacheline to arrive from the GPU. Typical throughput
 *     is 2-6 GB/s depending on PCIe gen and payload size.
 *   - memcpy_toio (host->GPU): PCIe posted writes. The CPU can fire and
 *     forget each write. Typically faster than reads, ~8-12 GB/s on
 *     PCIe Gen4 x16.
 */

/*
 * gpu_bar_memcpy_fromio() -- copy GPU VRAM -> host buffer, page-by-page
 *
 * Reads 'size' bytes from GPU BAR pages starting at 'offset' into
 * 'host_dst'. Each memcpy_fromio() call translates to PCIe MMIO read
 * transactions from the GPU's BAR1 aperture.
 *
 * @host_dst:  Destination in host DRAM (kernel virtual address).
 * @gpu_buf:   GPU buffer containing the bar_pages[] array.
 * @offset:    Byte offset into the GPU buffer to start reading from.
 * @size:      Number of bytes to read.
 */
static void gpu_bar_memcpy_fromio(void *host_dst,
				  struct dmaplane_gpu_buffer *gpu_buf,
				  u64 offset, u64 size)
{
	u64 remaining = size;
	u64 pos = 0;

	while (remaining > 0) {
		int page_idx = (offset + pos) / DMAPLANE_GPU_PAGE_SIZE;
		u64 page_off = (offset + pos) % DMAPLANE_GPU_PAGE_SIZE;
		u64 chunk = DMAPLANE_GPU_PAGE_SIZE - page_off;

		if (chunk > remaining)
			chunk = remaining;

		memcpy_fromio((char *)host_dst + pos,
			      gpu_buf->bar_pages[page_idx] + page_off,
			      chunk);

		pos += chunk;
		remaining -= chunk;
	}
}

/*
 * gpu_bar_memcpy_toio() -- copy host buffer -> GPU VRAM, page-by-page
 *
 * Writes 'size' bytes from 'host_src' into GPU BAR pages starting at
 * 'offset'. Each memcpy_toio() call translates to PCIe posted write
 * transactions to the GPU's BAR1 aperture.
 *
 * @gpu_buf:   GPU buffer containing the bar_pages[] array.
 * @offset:    Byte offset into the GPU buffer to start writing to.
 * @host_src:  Source in host DRAM (kernel virtual address).
 * @size:      Number of bytes to write.
 */
static void gpu_bar_memcpy_toio(struct dmaplane_gpu_buffer *gpu_buf,
				u64 offset, const void *host_src, u64 size)
{
	u64 remaining = size;
	u64 pos = 0;

	while (remaining > 0) {
		int page_idx = (offset + pos) / DMAPLANE_GPU_PAGE_SIZE;
		u64 page_off = (offset + pos) % DMAPLANE_GPU_PAGE_SIZE;
		u64 chunk = DMAPLANE_GPU_PAGE_SIZE - page_off;

		if (chunk > remaining)
			chunk = remaining;

		memcpy_toio(gpu_buf->bar_pages[page_idx] + page_off,
			    (const char *)host_src + pos,
			    chunk);

		pos += chunk;
		remaining -= chunk;
	}
}

/* ============================================================================
 * dmaplane_gpu_dma_to_host() -- GPU VRAM -> host DRAM transfer
 * ============================================================================
 *
 * Copies data from a pinned GPU buffer into a host DMA buffer. The
 * transfer is CPU-driven: the CPU reads from ioremap_wc'd BAR pages
 * (memcpy_fromio), pulling data across the PCIe bus.
 *
 * Locking strategy:
 *   - Takes gpu_buf_lock briefly to look up and validate the GPU buffer
 *   - Takes buf_lock briefly to look up and validate the host buffer
 *   - Releases BOTH locks before the actual memcpy -- the copy may take
 *     milliseconds for large buffers and we must not hold mutexes during
 *     that time (would block all other pin/unpin/buffer operations)
 *   - Checks gpu_revoked one final time right before the copy as a
 *     best-effort race guard (the revocation could still happen mid-copy,
 *     but the window is minimized)
 *
 * The host buffer's vaddr comes from vmap() in dmabuf_rdma.c for
 * BUF_TYPE_PAGES buffers, or from dma_alloc_coherent() for
 * BUF_TYPE_COHERENT buffers. Either way, it's a valid kernel VA.
 *
 * See gpu_p2p.h for full parameter documentation.
 */
int dmaplane_gpu_dma_to_host(struct dmaplane_dev *edev,
			     __u32 gpu_handle, __u32 host_handle,
			     u64 offset, u64 size, u64 *elapsed_ns)
{
	struct dmaplane_gpu_buffer *gpu_buf;
	struct dmaplane_buffer *host_buf;
	ktime_t start, end;
	u64 ns;

	/* Look up GPU buffer */
	mutex_lock(&edev->gpu_buf_lock);
	gpu_buf = dmaplane_gpu_find_buffer(edev, gpu_handle);
	if (!gpu_buf) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -ENOENT;
	}
	if (atomic_read(&gpu_buf->gpu_revoked)) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -ENODEV;
	}
	if (offset + size > gpu_buf->size) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -EINVAL;
	}
	mutex_unlock(&edev->gpu_buf_lock);

	/* Look up host buffer */
	mutex_lock(&edev->buf_lock);
	host_buf = dmabuf_rdma_find_buffer(edev, host_handle);
	if (!host_buf || !host_buf->vaddr) {
		mutex_unlock(&edev->buf_lock);
		return -ENOENT;
	}
	if (offset + size > host_buf->size) {
		mutex_unlock(&edev->buf_lock);
		return -EINVAL;
	}
	mutex_unlock(&edev->buf_lock);

	/* Best-effort revocation check right before the copy */
	if (atomic_read(&gpu_buf->gpu_revoked))
		return -ENODEV;

	start = ktime_get();
	gpu_bar_memcpy_fromio((char *)host_buf->vaddr + offset,
			      gpu_buf, offset, size);
	end = ktime_get();

	ns = ktime_to_ns(ktime_sub(end, start));
	if (elapsed_ns)
		*elapsed_ns = ns;

	/* Phase 8 instrumentation: tracepoint + stats */
	{
		u64 mbps = ns > 0 ? size * 1000 / ns : 0;
		trace_dmaplane_gpu_dma("g2h", size, ns, mbps);
	}
	atomic64_add(size, &edev->gpu_stats.dma_g2h_bytes);

	pr_debug("GPU->host DMA %llu bytes in %llu ns\n",
		 size, ns);
	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_gpu_dma_to_host);

/* ============================================================================
 * dmaplane_gpu_dma_from_host() -- host DRAM -> GPU VRAM transfer
 * ============================================================================
 *
 * Copies data from a host DMA buffer into a pinned GPU buffer. The
 * transfer is CPU-driven: the CPU writes to ioremap_wc'd BAR pages
 * (memcpy_toio), pushing data across the PCIe bus via posted writes.
 * Write-combining coalesces stores into full PCIe TLP payloads.
 *
 * PCIe posted writes (host->GPU) are typically faster than non-posted
 * reads (GPU->host) because the CPU doesn't wait for an acknowledgment
 * from the GPU for each write -- it can pipeline multiple writes in the
 * PCIe transaction layer.
 *
 * Locking and validation follow the same pattern as dma_to_host().
 *
 * See gpu_p2p.h for full parameter documentation.
 */
int dmaplane_gpu_dma_from_host(struct dmaplane_dev *edev,
			       __u32 host_handle, __u32 gpu_handle,
			       u64 offset, u64 size, u64 *elapsed_ns)
{
	struct dmaplane_gpu_buffer *gpu_buf;
	struct dmaplane_buffer *host_buf;
	ktime_t start, end;
	u64 ns;

	/* Look up GPU buffer */
	mutex_lock(&edev->gpu_buf_lock);
	gpu_buf = dmaplane_gpu_find_buffer(edev, gpu_handle);
	if (!gpu_buf) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -ENOENT;
	}
	if (atomic_read(&gpu_buf->gpu_revoked)) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -ENODEV;
	}
	if (offset + size > gpu_buf->size) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -EINVAL;
	}
	mutex_unlock(&edev->gpu_buf_lock);

	/* Look up host buffer */
	mutex_lock(&edev->buf_lock);
	host_buf = dmabuf_rdma_find_buffer(edev, host_handle);
	if (!host_buf || !host_buf->vaddr) {
		mutex_unlock(&edev->buf_lock);
		return -ENOENT;
	}
	if (offset + size > host_buf->size) {
		mutex_unlock(&edev->buf_lock);
		return -EINVAL;
	}
	mutex_unlock(&edev->buf_lock);

	/* Best-effort revocation check right before the copy */
	if (atomic_read(&gpu_buf->gpu_revoked))
		return -ENODEV;

	start = ktime_get();
	gpu_bar_memcpy_toio(gpu_buf, offset,
			    (const char *)host_buf->vaddr + offset, size);
	end = ktime_get();

	ns = ktime_to_ns(ktime_sub(end, start));
	if (elapsed_ns)
		*elapsed_ns = ns;

	/* Phase 8 instrumentation: tracepoint + stats */
	{
		u64 mbps = ns > 0 ? size * 1000 / ns : 0;
		trace_dmaplane_gpu_dma("h2g", size, ns, mbps);
	}
	atomic64_add(size, &edev->gpu_stats.dma_h2g_bytes);

	pr_debug("host->GPU DMA %llu bytes in %llu ns\n",
		 size, ns);
	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_gpu_dma_from_host);

/* ============================================================================
 * dmaplane_gpu_benchmark() -- measure host<->GPU BAR throughput
 * ============================================================================
 *
 * Runs a bidirectional bandwidth benchmark on pre-pinned GPU and host
 * buffers. Transfers the full buffer (min of GPU and host sizes) for
 * the specified number of iterations in each direction, measuring
 * wall-clock time via ktime_get() and computing throughput in MB/s.
 *
 * Benchmark methodology:
 *   - Host->GPU: runs 'iterations' full-buffer memcpy_toio transfers
 *     back-to-back, measures total time, computes MB/s.
 *   - GPU->Host: same, with memcpy_fromio.
 *   - Each direction is measured independently (not interleaved).
 *   - Revocation is checked before each iteration. If the GPU pages
 *     are revoked mid-benchmark, returns -ENODEV immediately.
 *
 * Throughput formula:
 *   MB/s = (transfer_size * iterations * 1000) / total_nanoseconds
 *   This works because: bytes/ns = GB/s, and GB/s * 1000 = MB/s.
 *   (More precisely: bytes * 10^9 / ns / 10^6 = bytes * 10^3 / ns)
 *
 * This is a kernel-only benchmark -- the actual pin/unpin latency is
 * measured by userspace by wrapping ktime around the IOCTL_GPU_PIN
 * and IOCTL_GPU_UNPIN calls. This function only measures the data
 * transfer throughput.
 *
 * Note: the benchmark runs at softirq-disabled priority (ktime_get
 * disables preemption briefly). For large transfers or many iterations,
 * this may cause scheduling latency for other tasks. Userspace should
 * choose iterations appropriately.
 *
 * See gpu_p2p.h for full parameter documentation.
 */
int dmaplane_gpu_benchmark(struct dmaplane_dev *edev,
			   __u32 gpu_handle, __u32 host_handle,
			   __u32 iterations, u64 *h2g_mbps, u64 *g2h_mbps)
{
	struct dmaplane_gpu_buffer *gpu_buf;
	struct dmaplane_buffer *host_buf;
	ktime_t start, end;
	u64 total_ns;
	u64 transfer_size;
	unsigned int i;

	if (iterations == 0)
		return -EINVAL;

	/* Validate GPU buffer */
	mutex_lock(&edev->gpu_buf_lock);
	gpu_buf = dmaplane_gpu_find_buffer(edev, gpu_handle);
	if (!gpu_buf) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -ENOENT;
	}
	if (atomic_read(&gpu_buf->gpu_revoked)) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -ENODEV;
	}
	transfer_size = gpu_buf->size;
	mutex_unlock(&edev->gpu_buf_lock);

	/* Validate host buffer -- use the smaller of the two sizes */
	mutex_lock(&edev->buf_lock);
	host_buf = dmabuf_rdma_find_buffer(edev, host_handle);
	if (!host_buf || !host_buf->vaddr) {
		mutex_unlock(&edev->buf_lock);
		return -ENOENT;
	}
	if (host_buf->size < transfer_size)
		transfer_size = host_buf->size;
	mutex_unlock(&edev->buf_lock);

	if (transfer_size == 0)
		return -EINVAL;

	/* Benchmark host->GPU (memcpy_toio -- PCIe posted writes) */
	start = ktime_get();
	for (i = 0; i < iterations; i++) {
		if (atomic_read(&gpu_buf->gpu_revoked))
			return -ENODEV;
		gpu_bar_memcpy_toio(gpu_buf, 0, host_buf->vaddr,
				    transfer_size);
		cond_resched();
	}
	end = ktime_get();
	total_ns = ktime_to_ns(ktime_sub(end, start));
	if (total_ns > 0) {
		u64 total_bytes = transfer_size * iterations;
		*h2g_mbps = total_bytes * 1000 / total_ns;
	} else {
		*h2g_mbps = 0;
	}

	/* Benchmark GPU->host (memcpy_fromio -- PCIe non-posted reads) */
	start = ktime_get();
	for (i = 0; i < iterations; i++) {
		if (atomic_read(&gpu_buf->gpu_revoked))
			return -ENODEV;
		gpu_bar_memcpy_fromio(host_buf->vaddr, gpu_buf, 0,
				      transfer_size);
		cond_resched();
	}
	end = ktime_get();
	total_ns = ktime_to_ns(ktime_sub(end, start));
	if (total_ns > 0) {
		u64 total_bytes = transfer_size * iterations;
		*g2h_mbps = total_bytes * 1000 / total_ns;
	} else {
		*g2h_mbps = 0;
	}

	pr_info("GPU benchmark %u iters x %llu bytes: h2g=%llu MB/s g2h=%llu MB/s\n",
		iterations, transfer_size, *h2g_mbps, *g2h_mbps);
	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_gpu_benchmark);

/* ============================================================================
 * dmaplane_gpu_register_mr() -- register GPU BAR pages as an RDMA MR
 * ============================================================================
 *
 * Creates a Memory Region backed by GPU BAR pages so rxe can send
 * GPU VRAM contents over the network. Uses the contiguous WC mapping
 * (rdma_vaddr) as sge.addr -- rxe interprets this as a kernel virtual
 * address and does memcpy from it.
 *
 * The MR is stored in the same mrs[] array as host-backed MRs, using
 * the same lock, ID namespace, and deregister path. The existing
 * rdma_engine_deregister_mr() handles cleanup -- it frees the SG table
 * and clears the slot. The ib_dma_unmap_sg call in deregister is a
 * no-op for rxe (software RDMA has no real IOMMU).
 *
 * A separate SG table is built (not shared with gpu_buf->sgt) because
 * the deregister path frees the MR's SG table.
 *
 * Caller must hold edev->rdma_sem (read lock).
 */
int dmaplane_gpu_register_mr(struct dmaplane_dev *edev,
			     struct dmaplane_gpu_mr_params *params)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct dmaplane_gpu_buffer *gpu_buf;
	struct dmaplane_mr_entry *mr_entry;
	struct sg_table *sgt;
	struct scatterlist *sg;
	ktime_t start;
	int i;

	if (!ctx->initialized)
		return -EINVAL;

	/* Look up the GPU buffer and validate preconditions:
	 *   1. Buffer exists and is in_use
	 *   2. Not revoked (NVIDIA hasn't reclaimed the pages)
	 *   3. Has a contiguous RDMA mapping (BAR pages were contiguous) */
	mutex_lock(&edev->gpu_buf_lock);
	gpu_buf = NULL;
	for (i = 0; i < DMAPLANE_MAX_GPU_BUFFERS; i++) {
		if (edev->gpu_buffers[i].in_use &&
		    edev->gpu_buffers[i].id == params->gpu_handle) {
			gpu_buf = &edev->gpu_buffers[i];
			break;
		}
	}
	if (!gpu_buf) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -ENOENT;
	}
	if (atomic_read(&gpu_buf->gpu_revoked)) {
		mutex_unlock(&edev->gpu_buf_lock);
		return -ENODEV;
	}
	if (!gpu_buf->rdma_vaddr) {
		mutex_unlock(&edev->gpu_buf_lock);
		pr_err("GPU buffer %u has no contiguous RDMA mapping\n",
		       params->gpu_handle);
		return -EINVAL;
	}
	mutex_unlock(&edev->gpu_buf_lock);

	/* Build a SEPARATE SG table for this MR (not shared with
	 * gpu_buf->sgt). The deregister path (rdma_engine_deregister_mr)
	 * frees the MR's SG table -- if we shared gpu_buf->sgt, unpin
	 * would double-free it. */
	sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	if (!sgt)
		return -ENOMEM;
	if (sg_alloc_table(sgt, gpu_buf->num_pages, GFP_KERNEL)) {
		kfree(sgt);
		return -ENOMEM;
	}

	/* Fill SG entries with GPU BAR physical addresses directly.
	 *
	 * Normal host MRs call ib_dma_map_sg() to get IOMMU-translated
	 * DMA addresses. GPU BAR pages have no struct page, so we cannot
	 * use the DMA mapping API. Instead, we set sg_dma_address directly
	 * to the BAR1 physical address -- this works for rxe (software RDMA
	 * with no real IOMMU) but would need nvidia_p2p_dma_map_pages()
	 * for hardware NICs like ConnectX. */
	i = 0;
	for_each_sg(sgt->sgl, sg, gpu_buf->num_pages, i) {
		sg_dma_address(sg) = gpu_buf->page_table->pages[i]->physical_address;
		sg_dma_len(sg) = DMAPLANE_GPU_PAGE_SIZE;
	}

	/* Find a free MR slot in the shared mrs[] array.
	 * GPU and host MRs coexist in the same ID namespace -- the
	 * existing deregister and lookup paths work unchanged. */
	start = ktime_get();
	mutex_lock(&edev->mr_lock);
	mr_entry = NULL;
	for (i = 0; i < DMAPLANE_MAX_MRS; i++) {
		if (!edev->mrs[i].in_use) {
			mr_entry = &edev->mrs[i];
			break;
		}
	}
	if (!mr_entry) {
		mutex_unlock(&edev->mr_lock);
		sg_free_table(sgt);
		kfree(sgt);
		return -ENOMEM;
	}

	/* Populate the MR entry.
	 *
	 * Key differences from host MR registration:
	 *   - buf_id = 0: no host buffer association (GPU-backed)
	 *   - mr = NULL: no ib_alloc_mr / ib_map_mr_sg -- we use
	 *     local_dma_lkey which authorizes access to all DMA-mapped
	 *     addresses without a per-MR hardware key
	 *   - rkey = 0: local_dma_lkey is local-only (no remote access)
	 *   - sge_addr = rdma_vaddr: the contiguous WC kernel VA that
	 *     rxe will memcpy from when processing send WRs */
	mr_entry->id = edev->next_mr_id;
	if (++edev->next_mr_id == 0)
		edev->next_mr_id = 1;
	mr_entry->buf_id = 0;
	mr_entry->mr = NULL;
	mr_entry->lkey = ctx->pd->local_dma_lkey;
	mr_entry->rkey = 0;
	mr_entry->sge_addr = (u64)(uintptr_t)gpu_buf->rdma_vaddr;
	mr_entry->sgt = sgt;
	mr_entry->sgt_nents = sgt->nents;
	mr_entry->in_use = true;
	mr_entry->reg_time = ktime_sub(ktime_get(), start);

	params->mr_id = mr_entry->id;
	params->lkey = mr_entry->lkey;
	params->rkey = mr_entry->rkey;

	atomic64_inc(&edev->stats.mrs_registered);
	mutex_unlock(&edev->mr_lock);

	/* Phase 8 instrumentation: track GPU MR registrations separately */
	atomic64_inc(&edev->gpu_stats.gpu_mrs_registered);

	pr_info("GPU MR %u registered (gpu_handle=%u, %d pages, lkey=0x%x)\n",
		params->mr_id, params->gpu_handle,
		gpu_buf->num_pages, params->lkey);
	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_gpu_register_mr);

/* ============================================================================
 * benchmark_gpu_loopback() -- RDMA loopback from GPU MR to host MR
 * ============================================================================
 *
 * Sends data from a GPU-backed MR (on QP-A) to a host-backed MR
 * (on QP-B) via the rxe loopback QP pair. This proves that rxe can
 * memcpy from the GPU BAR's contiguous WC mapping -- the critical
 * step that enables GPU VRAM to be sent over real RDMA networks.
 *
 * Follows the same pattern as benchmark_loopback() in benchmark.c:
 * post recv first, post send, poll both CQs, measure latency.
 *
 * Caller must hold edev->rdma_sem (read lock).
 */
#define GPU_LOOPBACK_TIMEOUT_MS 5000

int benchmark_gpu_loopback(struct dmaplane_dev *edev,
			   struct dmaplane_gpu_loopback_params *params)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct dmaplane_mr_entry gpu_mr, host_mr;
	struct poll_cq_wait send_wait, recv_wait;
	struct ib_wc wc;
	ktime_t start, end;
	int ret, rc;

	if (!ctx->initialized)
		return -EINVAL;

	if (params->size == 0)
		return -EINVAL;

	/* Snapshot both MR entries under lock.
	 *
	 * We copy the full mr_entry structs by value so the loopback
	 * can proceed without holding mr_lock during the actual RDMA
	 * operation. The snapshot captures sge_addr and lkey -- the only
	 * fields post_send/post_recv need. */
	mutex_lock(&edev->mr_lock);
	{
		struct dmaplane_mr_entry *gm, *hm;

		gm = dmabuf_rdma_find_mr(edev, params->gpu_mr_id);
		hm = dmabuf_rdma_find_mr(edev, params->host_mr_id);
		if (!gm || !hm) {
			mutex_unlock(&edev->mr_lock);
			return -ENOENT;
		}
		gpu_mr = *gm;
		host_mr = *hm;
	}
	mutex_unlock(&edev->mr_lock);

	start = ktime_get();

	/* Post recv FIRST on QP-B (host MR as receive buffer).
	 *
	 * RDMA requires the receive to be posted before the send --
	 * if the send completes before a recv is posted, the message
	 * is dropped and rxe reports a receiver-not-ready error. */
	ret = rdma_engine_post_recv(edev, ctx->qp_b, &host_mr,
				    params->size, &recv_wait);
	if (ret) {
		pr_err("GPU loopback recv post failed: %d\n", ret);
		return ret;
	}

	/* Post send on QP-A (GPU MR as send buffer).
	 *
	 * rxe processes the send by doing memcpy(dest, sge.addr, len)
	 * where sge.addr = gpu_buf->rdma_vaddr -- the contiguous WC
	 * mapping of GPU BAR pages. This is the critical path that
	 * proves GPU VRAM can be read via the RDMA subsystem. */
	ret = rdma_engine_post_send(edev, ctx->qp_a, &gpu_mr,
				    params->size, &send_wait);
	if (ret) {
		pr_err("GPU loopback send post failed: %d\n", ret);
		return ret;
	}

	/* Wait for send completion on CQ-A.
	 * 5s timeout covers worst-case for large buffers read from BAR
	 * (PCIe non-posted reads at ~100 MB/s -> 64KB takes <1ms). */
	rc = rdma_engine_poll_cq(ctx->cq_a, &wc, GPU_LOOPBACK_TIMEOUT_MS);
	if (rc <= 0) {
		pr_err("GPU loopback send poll %s\n",
		       rc == 0 ? "timed out" : "failed");
		return rc == 0 ? -ETIMEDOUT : rc;
	}
	if (wc.status != IB_WC_SUCCESS) {
		pr_err("GPU loopback send completion error: %s\n",
		       ib_wc_status_msg(wc.status));
		atomic64_inc(&edev->stats.completion_errors);
		return -EIO;
	}
	atomic64_inc(&edev->stats.completions_polled);
	atomic64_add(params->size, &edev->stats.bytes_sent);

	/* Wait for recv completion on CQ-B.
	 * wc.byte_len tells us how many bytes the receiver actually got --
	 * should match params->size for a successful loopback. */
	rc = rdma_engine_poll_cq(ctx->cq_b, &wc, GPU_LOOPBACK_TIMEOUT_MS);
	if (rc <= 0) {
		pr_err("GPU loopback recv poll %s\n",
		       rc == 0 ? "timed out" : "failed");
		return rc == 0 ? -ETIMEDOUT : rc;
	}
	if (wc.status != IB_WC_SUCCESS) {
		pr_err("GPU loopback recv completion error: %s\n",
		       ib_wc_status_msg(wc.status));
		atomic64_inc(&edev->stats.completion_errors);
		return -EIO;
	}
	atomic64_inc(&edev->stats.completions_polled);
	atomic64_add(wc.byte_len, &edev->stats.bytes_received);

	end = ktime_get();

	/* Populate output fields -- latency covers the full round trip
	 * (post recv -> post send -> poll send CQ -> poll recv CQ). */
	params->latency_ns = ktime_to_ns(ktime_sub(end, start));
	params->recv_bytes = wc.byte_len;
	params->status = 0;

	pr_info("GPU loopback %u bytes in %llu ns\n",
		params->recv_bytes, params->latency_ns);
	return 0;
}
EXPORT_SYMBOL_GPL(benchmark_gpu_loopback);

/* ============================================================================
 * dmaplane_gpu_init() -- resolve NVIDIA P2P symbols at runtime
 * ============================================================================
 *
 * Called from dmaplane_init() during module load when CONFIG_DMAPLANE_GPU
 * is enabled. Resolves the three nvidia_p2p_* function symbols from the
 * running kernel's symbol table using symbol_get().
 *
 * symbol_get() performs a runtime lookup in the kernel symbol table and
 * increments the owning module's refcount (preventing nvidia.ko from
 * being unloaded while dmaplane holds references). Unlike static symbol
 * resolution (which happens at modpost and insmod time), symbol_get()
 * does not create a module dependency in the depmod database -- it's a
 * purely runtime mechanism.
 *
 * Why this is necessary:
 *   nvidia.ko exports nvidia_p2p_* as EXPORT_SYMBOL_GPL, yet nvidia.ko
 *   itself is a proprietary (non-GPL) module. On kernel 6.5+, the
 *   module loader enforces a rule: a module that imports GPL-only symbols
 *   (from any source) cannot also depend on proprietary modules. Since
 *   dmaplane uses GPL-only symbols from ib_core (RDMA verbs), a static
 *   dependency on nvidia.ko causes insmod to fail with:
 *     "module using GPL-only symbols uses symbols from proprietary module"
 *   symbol_get() sidesteps this by avoiding the static dependency entirely.
 *
 * The function pointer casts are necessary because symbol_get() returns
 * void * (or rather, the symbol's address as a generic pointer), while
 * our typedefs have the full function signature.
 *
 * If any symbol cannot be resolved (nvidia.ko not loaded, or the driver
 * version doesn't export these symbols), all successfully resolved
 * symbols are released via dmaplane_gpu_exit() before returning -ENOENT.
 *
 * Return: 0 on success, -ENOENT if nvidia.ko is not loaded or symbols
 *         are not available.
 */
int dmaplane_gpu_init(void)
{
	nv_get_pages = (nv_p2p_get_pages_fn_t)symbol_get(nvidia_p2p_get_pages);
	nv_put_pages = (nv_p2p_put_pages_fn_t)symbol_get(nvidia_p2p_put_pages);
	nv_free_page_table = (nv_p2p_free_page_table_fn_t)symbol_get(nvidia_p2p_free_page_table);

	if (!nv_get_pages || !nv_put_pages || !nv_free_page_table) {
		pr_err("failed to resolve NVIDIA P2P symbols -- is nvidia.ko loaded?\n");
		dmaplane_gpu_exit();
		return -ENOENT;
	}

	pr_info("GPU P2P symbols resolved via symbol_get()\n");
	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_gpu_init);

/* ============================================================================
 * dmaplane_gpu_exit() -- release NVIDIA P2P symbol references
 * ============================================================================
 *
 * Called from dmaplane_exit() during module unload. Releases the module
 * references that were acquired by symbol_get() in dmaplane_gpu_init().
 *
 * symbol_put() decrements the owning module's refcount. Once all
 * references are released, nvidia.ko becomes eligible for unloading
 * (if no other consumers remain).
 *
 * Each symbol is checked for non-NULL before calling symbol_put(),
 * making this safe to call even if dmaplane_gpu_init() partially
 * failed (e.g. nvidia_p2p_get_pages resolved but nvidia_p2p_put_pages
 * did not). After releasing, each pointer is set to NULL to prevent
 * double-put.
 *
 * Note: symbol_put() takes a symbol NAME (it's a macro that expands
 * to a module_put on the module containing that symbol), not the
 * function pointer value.
 */
void dmaplane_gpu_exit(void)
{
	if (nv_get_pages) {
		symbol_put(nvidia_p2p_get_pages);
		nv_get_pages = NULL;
	}
	if (nv_put_pages) {
		symbol_put(nvidia_p2p_put_pages);
		nv_put_pages = NULL;
	}
	if (nv_free_page_table) {
		symbol_put(nvidia_p2p_free_page_table);
		nv_free_page_table = NULL;
	}
}
EXPORT_SYMBOL_GPL(dmaplane_gpu_exit);
