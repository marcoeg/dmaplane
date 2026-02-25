/* SPDX-License-Identifier: GPL-2.0 */
/*
 * gpu_p2p.h — GPU peer-to-peer memory interface for dmaplane
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Declares the data structures and functions for pinning GPU VRAM via
 * NVIDIA's P2P API and performing host<->GPU data transfers through
 * ioremap'd BAR1 pages.
 *
 * Architecture overview:
 *
 *   CUDA app                dmaplane.ko               nvidia.ko
 *   --------                -----------               ---------
 *   cudaMalloc() -->  IOCTL_GPU_PIN -->  nvidia_p2p_get_pages()
 *        |                  |                    |
 *        |                  |              BAR1 phys addrs
 *        |                  |              (64KB pages each)
 *        |                  |                    |
 *        |              ioremap() per page <-----+
 *        |              build sg_table
 *        |                  |
 *        |            memcpy_toio / memcpy_fromio
 *        |            (host<->GPU via BAR MMIO)
 *        |                  |
 *   cudaFree() --->  unpin callback -->  atomic_set(revoked)
 *        |                  |
 *        |            IOCTL_GPU_UNPIN -->  iounmap + free
 *
 * Key design constraints:
 *
 *   1. BAR pages are NOT host DRAM — they are PCIe MMIO windows.
 *      pfn_to_page() is undefined on these addresses. CPU access
 *      requires ioremap(), reads use memcpy_fromio(), writes use
 *      memcpy_toio().
 *
 *   2. BAR pages are NOT contiguous — each nvidia_p2p_page_t may
 *      have a physical_address at an arbitrary BAR1 offset. We
 *      ioremap each 64KB page individually and store an array of
 *      void __iomem * pointers.
 *
 *   3. Asynchronous revocation — the NVIDIA driver can reclaim pinned
 *      pages at any time (e.g. cudaFree from userspace). The unpin
 *      callback fires with NVIDIA locks held. It MUST be trivial:
 *      set an atomic flag and return. No locks, no printk, no
 *      blocking.
 *
 *   4. Symbol resolution — nvidia.ko is a proprietary module. Kernel
 *      6.5+ refuses to let GPL modules statically link against it.
 *      All nvidia_p2p_* functions are resolved at runtime via
 *      symbol_get() in dmaplane_gpu_init().
 *
 * This header is guarded by CONFIG_DMAPLANE_GPU, which the Makefile
 * sets only when NVIDIA P2P headers are found at build time. When
 * the GPU subsystem is not compiled in, dmaplane.h uses only a
 * forward declaration of struct dmaplane_gpu_buffer.
 */

#ifndef _DMAPLANE_GPU_P2P_H
#define _DMAPLANE_GPU_P2P_H

#include "dmaplane.h"

#ifdef CONFIG_DMAPLANE_GPU

#include <nv-p2p.h>

/*
 * GPU BAR pages are always 64KB regardless of the host PAGE_SIZE.
 * This is a hardware constant set by the NVIDIA GPU's BAR1 mapping
 * granularity. nvidia_p2p_get_pages() requires both the virtual
 * address and length to be aligned to this size.
 */
#define DMAPLANE_GPU_PAGE_SIZE	(64 * 1024)

/*
 * struct dmaplane_gpu_buffer — per-pinned-GPU-region tracking
 *
 * Each call to dmaplane_gpu_pin() allocates one of these from the
 * fixed-size gpu_buffers[] array in struct dmaplane_dev. The struct
 * tracks the full lifecycle of a pinned GPU VRAM region: from the
 * initial nvidia_p2p_get_pages() call through ioremap setup, DMA
 * operations, and eventual cleanup.
 *
 * Lifetime: allocated on pin, freed on unpin. The gpu_revoked flag
 * may be set asynchronously by the NVIDIA unpin callback at any point
 * between pin and unpin — all DMA functions check it before access.
 *
 * Locking: all fields except gpu_revoked and revoke_done are protected
 * by edev->gpu_buf_lock. gpu_revoked uses atomic_t because it is set
 * from the NVIDIA callback context where no dmaplane locks may be held.
 * revoke_done is a completion that callers can wait on if they need to
 * synchronize with revocation.
 *
 * @id:           Unique handle returned to userspace for subsequent
 *                ioctl operations (unpin, DMA, benchmark). Monotonically
 *                increasing, wraps at UINT_MAX, never reuses 0.
 *
 * @gpu_va:       CUDA virtual address passed to nvidia_p2p_get_pages().
 *                Must be 64KB-aligned. Stored for nvidia_p2p_put_pages()
 *                which requires the original VA to release the pin.
 *
 * @size:         Total size of the pinned region in bytes. Must be a
 *                multiple of 64KB. Used for bounds checking in DMA ops.
 *
 * @page_table:   Opaque page table returned by nvidia_p2p_get_pages().
 *                Contains entries[] count and pages[] array, where each
 *                pages[i]->physical_address is a BAR1 physical address.
 *                Owned by NVIDIA — freed via put_pages or free_page_table
 *                depending on revocation state.
 *
 * @sgt:          Scatterlist table built from BAR physical addresses.
 *                Each sg entry has sg_dma_address set to the BAR1 phys
 *                addr and sg_dma_len set to 64KB. Used for future
 *                GPUDirect RDMA NIC registration (a NIC can DMA directly
 *                from these BAR addresses). Allocated by dmaplane, freed
 *                on unpin.
 *
 * @bar_pages:    Array of num_pages ioremap'd kernel virtual addresses,
 *                one per 64KB BAR page. bar_pages[i] maps to
 *                page_table->pages[i]->physical_address. CPU access to
 *                GPU VRAM goes through these MMIO mappings. Each entry
 *                is ioremap'd individually because BAR pages are not
 *                guaranteed physically contiguous.
 *
 * @rdma_vaddr:   Single contiguous WC mapping spanning all BAR pages.
 *                Created only if BAR pages are physically contiguous
 *                (typical for a single cudaMalloc). Used as sge.addr
 *                for RDMA MR registration — rxe's memcpy needs a
 *                contiguous kernel VA. NULL if pages are non-contiguous
 *                or if the contiguous ioremap_wc failed.
 *
 * @num_pages:    Number of 64KB pages in the pinned region. Equal to
 *                page_table->entries after a successful pin. Also the
 *                size of the bar_pages[] array.
 *
 * @gpu_revoked:  Atomic flag set to 1 by the unpin callback when NVIDIA
 *                reclaims the pages. Once set, all BAR page mappings are
 *                invalid — DMA functions return -ENODEV. Checked with
 *                atomic_read() before every BAR access. Never cleared
 *                once set (buffer must be unpinned and re-pinned).
 *
 * @revoke_done:  Completion signaled by the unpin callback after setting
 *                gpu_revoked. Allows synchronous waiters to block until
 *                revocation is complete (not currently used but available
 *                for future GPU<->RDMA teardown coordination).
 *
 * @numa_node:    NUMA node of the GPU, obtained via dev_to_node() on the
 *                platform device. Used by userspace for NUMA-aware host
 *                buffer placement (allocate host buffer on the same node
 *                as the GPU for optimal PCIe locality).
 *
 * @in_use:       Slot occupancy flag. True from pin to unpin. Protected
 *                by edev->gpu_buf_lock.
 */
struct dmaplane_gpu_buffer {
	__u32                       id;             /* unique handle for userspace */
	u64                         gpu_va;         /* CUDA VA passed to get_pages */
	u64                         size;           /* pinned region size in bytes */
	nvidia_p2p_page_table_t     *page_table;    /* NVIDIA page table (opaque) */
	struct sg_table             *sgt;           /* scatterlist from BAR addrs */
	void __iomem                **bar_pages;    /* per-page ioremap array */
	void __iomem                *rdma_vaddr;    /* contiguous WC mapping for RDMA (NULL if non-contiguous) */
	int                         num_pages;      /* number of 64KB BAR pages */
	atomic_t                    gpu_revoked;    /* 1 = NVIDIA reclaimed pages */
	struct completion            revoke_done;   /* signaled after revocation */
	int                         numa_node;      /* GPU's NUMA node (-1 if unknown) */
	bool                        in_use;         /* slot occupied (protected by gpu_buf_lock) */
};

/*
 * dmaplane_gpu_pin() — pin GPU VRAM and prepare BAR mappings
 *
 * Pins a contiguous region of GPU virtual memory via the NVIDIA P2P
 * API, builds a scatterlist from the returned BAR1 physical addresses,
 * and ioremap's each 64KB page for CPU access.
 *
 * @edev:             Device context (contains gpu_buffers[] and gpu_buf_lock)
 * @gpu_va:           CUDA virtual address to pin. Must be 64KB-aligned.
 *                    Obtained from cudaMalloc() in userspace.
 * @size:             Bytes to pin. Must be a positive 64KB multiple.
 * @handle_out:       Returns the buffer handle for subsequent operations.
 * @numa_node_out:    Returns the GPU's NUMA node (-1 if unknown).
 * @num_pages_out:    Returns the number of 64KB pages pinned.
 * @bar1_consumed_out: Returns total BAR1 bytes consumed (pages * 64KB).
 *
 * Locking: takes edev->gpu_buf_lock (mutex, may sleep). Held across
 * nvidia_p2p_get_pages() which can sleep — acceptable for this slow-path
 * pin operation. The lock protects the slot table only.
 *
 * Return: 0 on success, negative errno on failure:
 *   -EINVAL   gpu_va or size not 64KB-aligned, or size is 0
 *   -ENOMEM   no free GPU buffer slots, or allocation failure
 *   negative errno from nvidia_p2p_get_pages (platform-specific)
 */
int dmaplane_gpu_pin(struct dmaplane_dev *edev, u64 gpu_va, u64 size,
		     __u32 *handle_out, __s32 *numa_node_out,
		     __u32 *num_pages_out, u64 *bar1_consumed_out);

/*
 * dmaplane_gpu_unpin() — release a pinned GPU buffer
 *
 * Tears down all resources associated with a pinned GPU buffer:
 * iounmap's each BAR page, frees the scatterlist, and releases the
 * NVIDIA page table. Handles both normal and revoked states:
 *
 *   - Normal (gpu_revoked == 0): calls nvidia_p2p_put_pages() to
 *     release the pin and free the page table in one step.
 *
 *   - Revoked (gpu_revoked == 1): the NVIDIA unpin callback already
 *     fired. Calling put_pages would deadlock because NVIDIA holds
 *     internal locks. Instead calls nvidia_p2p_free_page_table() to
 *     free just the page_table struct.
 *
 * @edev:    Device context.
 * @handle:  Buffer handle from dmaplane_gpu_pin().
 *
 * Locking: takes edev->gpu_buf_lock.
 *
 * Return: 0 on success, -ENOENT if handle not found.
 */
int dmaplane_gpu_unpin(struct dmaplane_dev *edev, __u32 handle);

/*
 * dmaplane_gpu_dma_to_host() — GPU VRAM -> host DRAM transfer
 *
 * Copies data from GPU BAR pages into a host DMA buffer using
 * memcpy_fromio(). Each CPU load crosses the PCIe bus as an MMIO
 * read from the GPU's BAR1 aperture.
 *
 * The transfer walks the bar_pages[] array page-by-page, handling
 * 64KB page boundary crossings with per-page offset arithmetic.
 *
 * @edev:        Device context.
 * @gpu_handle:  GPU buffer handle (source).
 * @host_handle: Host buffer handle (destination). Must have a valid
 *               vaddr (BUF_TYPE_PAGES buffers get one from vmap()).
 * @offset:      Byte offset into both buffers.
 * @size:        Bytes to transfer.
 * @elapsed_ns:  If non-NULL, returns wall-clock nanoseconds for the copy.
 *
 * Locking: takes gpu_buf_lock and buf_lock briefly for lookup, then
 * releases both before the actual memcpy (which may take milliseconds
 * for large transfers).
 *
 * Return: 0 on success, -ENOENT if either buffer not found, -ENODEV
 *         if GPU pages were revoked, -EINVAL if offset+size exceeds
 *         buffer bounds.
 */
int dmaplane_gpu_dma_to_host(struct dmaplane_dev *edev,
			     __u32 gpu_handle, __u32 host_handle,
			     u64 offset, u64 size, u64 *elapsed_ns);

/*
 * dmaplane_gpu_dma_from_host() — host DRAM -> GPU VRAM transfer
 *
 * Copies data from a host DMA buffer into GPU BAR pages using
 * memcpy_toio(). Each CPU store crosses the PCIe bus as an MMIO
 * write to the GPU's BAR1 aperture. PCIe posted writes make this
 * direction typically faster than GPU->host reads.
 *
 * @edev:        Device context.
 * @host_handle: Host buffer handle (source).
 * @gpu_handle:  GPU buffer handle (destination).
 * @offset:      Byte offset into both buffers.
 * @size:        Bytes to transfer.
 * @elapsed_ns:  If non-NULL, returns wall-clock nanoseconds for the copy.
 *
 * Locking: same as dma_to_host — brief lock for lookup, unlocked for copy.
 *
 * Return: same error codes as dma_to_host.
 */
int dmaplane_gpu_dma_from_host(struct dmaplane_dev *edev,
			       __u32 host_handle, __u32 gpu_handle,
			       u64 offset, u64 size, u64 *elapsed_ns);

/*
 * dmaplane_gpu_benchmark() — measure host<->GPU BAR throughput
 *
 * Runs repeated memcpy transfers in both directions between a
 * pre-pinned GPU buffer and a pre-allocated host buffer, measuring
 * wall-clock time and reporting throughput in MB/s.
 *
 * This is a kernel-only benchmark invoked via ioctl. Pin/unpin
 * latency is measured by userspace separately (ktime around the
 * individual IOCTL_GPU_PIN/UNPIN calls).
 *
 * Transfer size is min(gpu_buf->size, host_buf->size). The full
 * buffer is transferred each iteration.
 *
 * @edev:        Device context.
 * @gpu_handle:  GPU buffer handle (must be pinned and not revoked).
 * @host_handle: Host buffer handle (must have valid vaddr).
 * @iterations:  Number of full-buffer transfers per direction. Must be > 0.
 * @h2g_mbps:    Returns host->GPU throughput in MB/s.
 * @g2h_mbps:    Returns GPU->host throughput in MB/s.
 *
 * Return: 0 on success, -ENOENT/-ENODEV/-EINVAL per usual.
 */
int dmaplane_gpu_benchmark(struct dmaplane_dev *edev,
			   __u32 gpu_handle, __u32 host_handle,
			   __u32 iterations, u64 *h2g_mbps, u64 *g2h_mbps);

/*
 * dmaplane_gpu_register_mr() — register GPU BAR pages as an RDMA MR
 *
 * Creates a Memory Region backed by GPU BAR pages so rxe can send
 * GPU VRAM contents via RDMA loopback. Uses the contiguous WC mapping
 * (rdma_vaddr) as the sge.addr for rxe's memcpy.
 *
 * The MR goes in the same mrs[] array as host MRs, using the same
 * ID namespace. Existing rdma_engine_deregister_mr() handles cleanup.
 *
 * Requires: RDMA initialized, GPU buffer pinned and not revoked,
 * rdma_vaddr != NULL (BAR pages must be physically contiguous).
 *
 * @edev:    Device context.
 * @params:  GPU MR params (gpu_handle in, mr_id/lkey/rkey out).
 *
 * Return: 0 on success, -EINVAL/-ENODEV/-ENOMEM on error.
 */
int dmaplane_gpu_register_mr(struct dmaplane_dev *edev,
			     struct dmaplane_gpu_mr_params *params);

/*
 * benchmark_gpu_loopback() — RDMA loopback from GPU MR to host MR
 *
 * Sends data from a GPU-backed MR on QP-A and receives it into a
 * host-backed MR on QP-B. Follows the same pattern as benchmark_loopback()
 * but uses separate MRs for send (GPU) and recv (host).
 *
 * @edev:    Device context.
 * @params:  Loopback params (gpu_mr_id, host_mr_id, size in;
 *           latency_ns, recv_bytes, status out).
 *
 * Return: 0 on success, negative errno on failure.
 */
int benchmark_gpu_loopback(struct dmaplane_dev *edev,
			   struct dmaplane_gpu_loopback_params *params);

/*
 * dmaplane_gpu_find_buffer() — look up a GPU buffer by handle
 *
 * Linear scan of gpu_buffers[]. Caller must hold edev->gpu_buf_lock
 * or otherwise ensure the array is stable.
 *
 * @edev:    Device context.
 * @handle:  Buffer handle to find.
 *
 * Return: pointer to the buffer struct, or NULL if not found.
 */
struct dmaplane_gpu_buffer *dmaplane_gpu_find_buffer(struct dmaplane_dev *edev,
						     __u32 handle);

/*
 * dmaplane_gpu_init() — resolve NVIDIA P2P symbols at runtime
 *
 * Called from dmaplane_init() during module load. Uses symbol_get()
 * to look up nvidia_p2p_get_pages, nvidia_p2p_put_pages, and
 * nvidia_p2p_free_page_table from the loaded nvidia.ko module.
 *
 * This runtime resolution avoids creating a static module dependency
 * on nvidia.ko, which would fail on kernel 6.5+ because the kernel
 * refuses to let a GPL module that uses EXPORT_SYMBOL_GPL symbols
 * also import symbols from a proprietary module.
 *
 * Prerequisite: nvidia.ko must be loaded before dmaplane.ko.
 *
 * Return: 0 on success, -ENOENT if any symbol could not be resolved
 *         (nvidia.ko not loaded or wrong driver version).
 */
int dmaplane_gpu_init(void);

/*
 * dmaplane_gpu_exit() — release NVIDIA P2P symbol references
 *
 * Called from dmaplane_exit() during module unload. Releases the
 * module references acquired by symbol_get() via symbol_put(),
 * allowing nvidia.ko to be unloaded if no other consumers remain.
 *
 * Safe to call even if dmaplane_gpu_init() partially failed —
 * only releases symbols that were successfully resolved.
 */
void dmaplane_gpu_exit(void);

#endif /* CONFIG_DMAPLANE_GPU */
#endif /* _DMAPLANE_GPU_P2P_H */
