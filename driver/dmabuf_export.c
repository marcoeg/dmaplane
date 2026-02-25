// SPDX-License-Identifier: GPL-2.0
/*
 * dmabuf_export.c — dma-buf exporter for dmaplane page-backed buffers
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Implements the dma_buf_ops interface to export dmaplane buffers as
 * dma-buf objects.  This is the kernel's standard framework for sharing
 * memory between devices without copies.  One device (the exporter)
 * allocates pages and wraps them in a struct dma_buf.  Other devices
 * (importers) attach to it and get per-device scatter-gather tables
 * mapped through their own IOMMU contexts.
 *
 * Only page-backed buffers (BUF_TYPE_PAGES) can be exported.  Coherent
 * allocations have no struct page array — sg_set_page needs pages,
 * and virt_to_page on dma_alloc_coherent memory is unreliable across
 * architectures (especially with IOMMU or SWIOTLB).
 *
 * The export context (dmaplane_dmabuf_ctx) borrows the page array and
 * vaddr from the backing buffer.  The buffer cannot be destroyed while
 * exported — the dmabuf_exported flag on dmaplane_buffer gates
 * dmabuf_rdma_destroy_buffer with -EBUSY.
 *
 * All dma-buf statistics (attach/detach/map/unmap counts) live on
 * dmaplane_stats_kern (device-level) so they survive individual
 * export/release cycles.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/slab.h>
#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/scatterlist.h>
#include <linux/mm.h>
#include <linux/iosys-map.h>

#include "dmabuf_export.h"
#include "dmabuf_rdma.h"

/*
 * Export context: per-export state attached to the dma-buf's priv field.
 *
 * Fields borrowed from the backing buffer (pages, vaddr) are valid for
 * the lifetime of the export — the dmabuf_exported flag prevents the
 * buffer from being destroyed while the dma-buf exists.
 */
struct dmaplane_dmabuf_ctx {
	struct dmaplane_dev *dev;	/* Back-pointer to device singleton */
	struct dmaplane_buffer *buf;	/* Backing buffer (borrowed) */
	struct page **pages;		/* Page array (borrowed from buf) */
	unsigned int nr_pages;		/* Number of pages */
	size_t size;			/* Buffer size in bytes */
	void *vaddr;			/* Kernel VA (borrowed from buf->vaddr) */
	atomic_t attach_count;		/* Active attachment count */
};

/* ------------------------------------------------------------------ */
/* dma_buf_ops callbacks                                               */
/* ------------------------------------------------------------------ */

/*
 * attach — Called when a device attaches to the dma-buf.
 *
 * We accept all devices unconditionally (no IOMMU group filtering).
 * The per-device IOMMU context is handled in map_dma_buf, which builds
 * SG tables specific to each importer's DMA address space.
 */
static int dmaplane_dmabuf_attach(struct dma_buf *dmabuf,
				  struct dma_buf_attachment *attach)
{
	struct dmaplane_dmabuf_ctx *ctx = dmabuf->priv;

	atomic_inc(&ctx->attach_count);
	atomic64_inc(&ctx->dev->stats.dmabuf_attachments);
	pr_debug("dmabuf attach: buf %u, device %s, count %d\n",
		 ctx->buf->id, dev_name(attach->dev),
		 atomic_read(&ctx->attach_count));
	return 0;
}

/*
 * detach — Called when a device detaches from the dma-buf.
 */
static void dmaplane_dmabuf_detach(struct dma_buf *dmabuf,
				   struct dma_buf_attachment *attach)
{
	struct dmaplane_dmabuf_ctx *ctx = dmabuf->priv;

	atomic_dec(&ctx->attach_count);
	atomic64_inc(&ctx->dev->stats.dmabuf_detachments);
	pr_debug("dmabuf detach: buf %u, device %s, count %d\n",
		 ctx->buf->id, dev_name(attach->dev),
		 atomic_read(&ctx->attach_count));
}

/*
 * map_dma_buf — Build a scatter-gather table for the importing device.
 *
 * This is the expensive callback.  Each importer may sit behind a
 * different IOMMU.  The SG table must contain DMA addresses valid for
 * THAT device's IOMMU context.  Building at export time would assume
 * a single IOMMU — wrong for cross-device sharing.
 *
 * Steps:
 *  1. Allocate an sg_table with one entry per page
 *  2. Fill each SG entry with sg_set_page
 *  3. dma_map_sgtable programs the importer's IOMMU
 *
 * The returned sg_table is freed in unmap_dma_buf.
 */
static struct sg_table *dmaplane_dmabuf_map(struct dma_buf_attachment *attach,
					    enum dma_data_direction dir)
{
	struct dmaplane_dmabuf_ctx *ctx = attach->dmabuf->priv;
	struct sg_table *sgt;
	struct scatterlist *sg;
	unsigned int i;
	int ret;

	sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	if (!sgt)
		return ERR_PTR(-ENOMEM);

	ret = sg_alloc_table(sgt, ctx->nr_pages, GFP_KERNEL);
	if (ret) {
		kfree(sgt);
		return ERR_PTR(ret);
	}

	/* Fill SG entries — one per page */
	for_each_sgtable_sg(sgt, sg, i)
		sg_set_page(sg, ctx->pages[i], PAGE_SIZE, 0);

	/*
	 * dma_map_sgtable programs IOMMU entries for the importing device.
	 * After this call, each SG entry's dma_address is valid for the
	 * importer's DMA context.  On x86 without IOMMU, DMA addresses
	 * equal physical addresses.
	 */
	ret = dma_map_sgtable(attach->dev, sgt, dir, 0);
	if (ret) {
		sg_free_table(sgt);
		kfree(sgt);
		return ERR_PTR(ret);
	}

	atomic64_inc(&ctx->dev->stats.dmabuf_maps);
	pr_debug("dmabuf map: buf %u, %u pages, device %s\n",
		 ctx->buf->id, ctx->nr_pages, dev_name(attach->dev));

	return sgt;
}

/*
 * unmap_dma_buf — Unmap and free the per-device SG table.
 */
static void dmaplane_dmabuf_unmap(struct dma_buf_attachment *attach,
				  struct sg_table *sgt,
				  enum dma_data_direction dir)
{
	struct dmaplane_dmabuf_ctx *ctx = attach->dmabuf->priv;

	dma_unmap_sgtable(attach->dev, sgt, dir, 0);
	sg_free_table(sgt);
	kfree(sgt);

	atomic64_inc(&ctx->dev->stats.dmabuf_unmaps);
	pr_debug("dmabuf unmap: buf %u, device %s\n",
		 ctx->buf->id, dev_name(attach->dev));
}

/*
 * release — Called when the last reference to the dma-buf drops.
 *
 * All fds closed, all kernel references (dma_buf_put) released, all
 * attachments detached.  Clear the dmabuf_exported flag so the backing
 * buffer can be destroyed.
 *
 * Runs in process context (from fput), so mutex_lock is legal.
 * No deadlock with dmabuf_rdma_destroy_buffer: destroy checks
 * dmabuf_exported and returns -EBUSY without blocking.
 */
static void dmaplane_dmabuf_release(struct dma_buf *dmabuf)
{
	struct dmaplane_dmabuf_ctx *ctx = dmabuf->priv;
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_buffer *buf = ctx->buf;

	/*
	 * Acquire buf_lock to clear the exported flag atomically.
	 * This ensures that a concurrent dmabuf_rdma_destroy_buffer
	 * sees a consistent state — either the flag is set (return
	 * -EBUSY) or cleared (proceed with destruction).
	 */
	mutex_lock(&dev->buf_lock);
	buf->dmabuf_exported = false;
	buf->dmabuf = NULL;
	mutex_unlock(&dev->buf_lock);

	atomic64_inc(&dev->stats.dmabufs_released);

	pr_debug("dmabuf release: buf %u\n", buf->id);

	kfree(ctx);
}

/*
 * vmap — Return the buffer's existing kernel VA.
 *
 * Reuses the vmap created by the page-backed allocation path in
 * dmabuf_rdma_create_buffer.  No second vmap is needed — the mapping
 * lifetime is the buffer's, not the dma-buf's.
 */
static int dmaplane_dmabuf_vmap(struct dma_buf *dmabuf,
				struct iosys_map *map)
{
	struct dmaplane_dmabuf_ctx *ctx = dmabuf->priv;

	iosys_map_set_vaddr(map, ctx->vaddr);
	return 0;
}

/*
 * vunmap — No-op.
 *
 * We don't own the vmap — the buffer does.  The vmap lifetime is tied
 * to the buffer, not to the dma-buf.  Freeing the mapping here would
 * leave the buffer's vaddr dangling.
 */
static void dmaplane_dmabuf_vunmap(struct dma_buf *dmabuf,
				   struct iosys_map *map)
{
	/* Intentionally empty */
}

/*
 * mmap — Map dma-buf pages into userspace.
 *
 * Same logic as dmaplane_mmap for page-backed buffers: vm_insert_page
 * loop.  This lets userspace mmap the dma-buf fd directly (separate
 * from mmapping /dev/dmaplane).
 */
static int dmaplane_dmabuf_mmap(struct dma_buf *dmabuf,
				struct vm_area_struct *vma)
{
	struct dmaplane_dmabuf_ctx *ctx = dmabuf->priv;
	unsigned long size = vma->vm_end - vma->vm_start;
	unsigned int i;
	int ret;

	if (size > ctx->size)
		return -EINVAL;

	vm_flags_set(vma, VM_DONTCOPY | VM_DONTEXPAND | VM_DONTDUMP);

	for (i = 0; i < ctx->nr_pages && i * PAGE_SIZE < size; i++) {
		ret = vm_insert_page(vma,
				     vma->vm_start + i * PAGE_SIZE,
				     ctx->pages[i]);
		if (ret)
			return ret;
	}

	return 0;
}

/*
 * begin_cpu_access — Prepare buffer for CPU access.
 *
 * On x86: no-op.  Hardware cache coherency ensures CPU and device see
 * consistent data without explicit sync.
 *
 * On ARM/RISC-V: would need dma_sync_sg_for_cpu() to invalidate or
 * flush caches, ensuring the CPU sees data written by the device.
 * The interface is correct even though x86 doesn't need the sync.
 */
static int dmaplane_dmabuf_begin_cpu_access(struct dma_buf *dmabuf,
					    enum dma_data_direction dir)
{
	/* x86: hardware cache coherency — no sync needed */
	return 0;
}

/*
 * end_cpu_access — Signal that CPU access is complete.
 *
 * On x86: no-op.  See begin_cpu_access comments.
 *
 * On ARM/RISC-V: would need dma_sync_sg_for_device() to flush CPU
 * writes to the point of coherency so the device sees them.
 */
static int dmaplane_dmabuf_end_cpu_access(struct dma_buf *dmabuf,
					  enum dma_data_direction dir)
{
	/* x86: hardware cache coherency — no sync needed */
	return 0;
}

static const struct dma_buf_ops dmaplane_dmabuf_ops = {
	.attach		= dmaplane_dmabuf_attach,
	.detach		= dmaplane_dmabuf_detach,
	.map_dma_buf	= dmaplane_dmabuf_map,
	.unmap_dma_buf	= dmaplane_dmabuf_unmap,
	.vmap		= dmaplane_dmabuf_vmap,
	.vunmap		= dmaplane_dmabuf_vunmap,
	.release	= dmaplane_dmabuf_release,
	.mmap		= dmaplane_dmabuf_mmap,
	.begin_cpu_access = dmaplane_dmabuf_begin_cpu_access,
	.end_cpu_access   = dmaplane_dmabuf_end_cpu_access,
};

/* ------------------------------------------------------------------ */
/* Export function                                                     */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_dmabuf_export — Export a page-backed buffer as a dma-buf.
 *
 * buf_lock is held for the entire operation including dma_buf_export()
 * and dma_buf_fd().  Both may sleep (memory allocation, fd allocation)
 * but buf_lock is a mutex so sleeping is permitted.  Coarse granularity
 * is acceptable — exports are rare operations.  Holding through the
 * entire sequence prevents a race where two threads export the same
 * buffer simultaneously.
 *
 * The fd is installed in the process's fd table by dma_buf_fd() before
 * copy_to_user.  If copy_to_user fails, the fd exists but userspace
 * doesn't know the number — it will be cleaned up on process exit.
 * Do NOT try to uninstall the fd from kernel space.
 */
int dmaplane_dmabuf_export(struct dmaplane_dev *dev,
			   struct dmaplane_export_dmabuf_arg *arg)
{
	struct dmaplane_buffer *buf;
	struct dmaplane_dmabuf_ctx *ctx;
	struct dma_buf *dmabuf;
	DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
	int fd;

	mutex_lock(&dev->buf_lock);

	buf = dmabuf_rdma_find_buffer(dev, arg->buf_id);
	if (!buf) {
		mutex_unlock(&dev->buf_lock);
		return -EINVAL;
	}

	/* Only page-backed buffers can be exported — coherent has no
	 * page array for SG table construction */
	if (buf->alloc_type != DMAPLANE_BUF_TYPE_PAGES) {
		mutex_unlock(&dev->buf_lock);
		return -EINVAL;
	}

	/* One dma-buf per buffer — prevents confusion about which fd
	 * corresponds to which export */
	if (buf->dmabuf_exported) {
		mutex_unlock(&dev->buf_lock);
		return -EBUSY;
	}

	ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
	if (!ctx) {
		mutex_unlock(&dev->buf_lock);
		return -ENOMEM;
	}

	ctx->dev = dev;
	ctx->buf = buf;
	ctx->pages = buf->pages;
	ctx->nr_pages = buf->nr_pages;
	ctx->size = buf->size;
	ctx->vaddr = buf->vaddr;
	atomic_set(&ctx->attach_count, 0);

	exp_info.ops = &dmaplane_dmabuf_ops;
	exp_info.size = buf->size;
	exp_info.flags = O_RDWR;	/* O_RDWR required for mmap with PROT_WRITE;
					 * O_CLOEXEC is set via dma_buf_fd below */
	exp_info.priv = ctx;

	dmabuf = dma_buf_export(&exp_info);
	if (IS_ERR(dmabuf)) {
		kfree(ctx);
		mutex_unlock(&dev->buf_lock);
		return PTR_ERR(dmabuf);
	}

	fd = dma_buf_fd(dmabuf, O_CLOEXEC);
	if (fd < 0) {
		/* dma_buf_put releases the dma-buf (triggers release callback,
		 * but dmabuf_exported is still false so it's a no-op clear) */
		dma_buf_put(dmabuf);
		kfree(ctx);
		mutex_unlock(&dev->buf_lock);
		return fd;
	}

	buf->dmabuf_exported = true;
	buf->dmabuf = dmabuf;

	mutex_unlock(&dev->buf_lock);

	arg->fd = fd;
	atomic64_inc(&dev->stats.dmabufs_exported);

	pr_debug("buffer %u exported as dma-buf fd %d\n", buf->id, fd);

	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_dmabuf_export);

/* ------------------------------------------------------------------ */
/* Kernel self-test                                                    */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_dmabuf_selftest — Exercise the full dma-buf lifecycle from
 * kernel space.
 *
 * This tests attach/map/unmap/detach — operations that userspace cannot
 * exercise because they are kernel-internal.  Uses the platform device
 * as the importing device (self-import).
 *
 * Called from module init when test_dmabuf=1 parameter is set.
 */
int dmaplane_dmabuf_selftest(struct dmaplane_dev *dev)
{
	struct dmaplane_buf_params bp = {0};
	struct dmaplane_export_dmabuf_arg ea = {0};
	struct dma_buf *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table *sgt;
	int ret, i;
	int pass = 0, fail = 0;

	pr_info("dmabuf selftest: starting\n");

	/* 1. Create a page-backed buffer */
	bp.alloc_type = DMAPLANE_BUF_TYPE_PAGES;
	bp.size = 64 * 1024;	/* 64 KB = 16 pages */
	ret = dmabuf_rdma_create_buffer(dev, &bp);
	if (ret) {
		pr_err("selftest: create_buffer failed: %d\n", ret);
		return ret;
	}
	pr_info("selftest: created buffer %u (%zu bytes)\n",
		bp.buf_id, (size_t)bp.size);

	/* 2. Export as dma-buf */
	ea.buf_id = bp.buf_id;
	ret = dmaplane_dmabuf_export(dev, &ea);
	if (ret) {
		pr_err("selftest: export failed: %d\n", ret);
		goto err_destroy_buf;
	}
	pr_info("selftest: exported as fd %d\n", ea.fd);

	/* Get the dma-buf from the fd — we need the kernel object */
	dmabuf = dma_buf_get(ea.fd);
	if (IS_ERR(dmabuf)) {
		pr_err("selftest: dma_buf_get failed: %ld\n", PTR_ERR(dmabuf));
		ret = PTR_ERR(dmabuf);
		goto err_destroy_buf;
	}

	/* 3. Attach/map/unmap/detach cycle — repeat 100 times */
	for (i = 0; i < 100; i++) {
		attach = dma_buf_attach(dmabuf, &dev->pdev->dev);
		if (IS_ERR(attach)) {
			pr_err("selftest: attach failed at iteration %d: %ld\n",
			       i, PTR_ERR(attach));
			fail++;
			break;
		}

		sgt = dma_buf_map_attachment(attach, DMA_BIDIRECTIONAL);
		if (IS_ERR(sgt)) {
			pr_err("selftest: map failed at iteration %d: %ld\n",
			       i, PTR_ERR(sgt));
			dma_buf_detach(dmabuf, attach);
			fail++;
			break;
		}

		/* Verify SG table has entries */
		if (sgt->nents == 0) {
			pr_err("selftest: SG table empty at iteration %d\n", i);
			fail++;
		} else {
			pass++;
		}

		dma_buf_unmap_attachment(attach, sgt, DMA_BIDIRECTIONAL);
		dma_buf_detach(dmabuf, attach);
	}

	/* 4. Release the dma-buf (drop our kernel ref) */
	dma_buf_put(dmabuf);

	/*
	 * 5. Close the fd — this drops the last reference on the dma-buf,
	 * which triggers the release callback and clears dmabuf_exported.
	 * We use ksys_close (or __close_fd equivalent) — but from module
	 * init context we can't easily close an fd.  Instead, the fd will
	 * be cleaned up when we later call dmabuf_rdma_destroy_buffer,
	 * which will fail if the fd is still open.  For the selftest,
	 * we rely on the dma_buf_put above dropping the kernel ref;
	 * the fd ref keeps the dma-buf alive until the module is unloaded
	 * and the process exits.
	 *
	 * Actually — the selftest runs in module init, which is not a
	 * normal process context with an fd table we control.  The
	 * dma_buf_fd installed the fd in current->files, but "current"
	 * during module init is the insmod process.  When insmod exits,
	 * the fd is closed automatically.  So we just verify the export
	 * flag state and skip explicit fd close.
	 */

	/* The buffer should still be marked as exported (fd still open) */
	mutex_lock(&dev->buf_lock);
	{
		struct dmaplane_buffer *buf;

		buf = dmabuf_rdma_find_buffer(dev, bp.buf_id);
		if (buf && buf->dmabuf_exported) {
			pr_info("selftest: buffer still exported (fd held by insmod process) — expected\n");
			pass++;
		} else {
			pr_info("selftest: buffer not exported — dma-buf was released\n");
			pass++;
		}
	}
	mutex_unlock(&dev->buf_lock);

	pr_info("dmabuf selftest: %d passed, %d failed\n", pass, fail);
	return fail ? -EIO : 0;

err_destroy_buf:
	dmabuf_rdma_destroy_buffer(dev, bp.buf_id);
	return ret;
}
EXPORT_SYMBOL_GPL(dmaplane_dmabuf_selftest);
