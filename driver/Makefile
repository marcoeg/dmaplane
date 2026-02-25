obj-m := dmaplane.o
dmaplane-objs := main.o dmabuf_rdma.o dmabuf_export.o rdma_engine.o benchmark.o \
                 numa_topology.o flow_control.o dmaplane_trace.o \
                 dmaplane_histogram.o dmaplane_debugfs.o

# -I$(src) is required for TRACE_INCLUDE_PATH = . to resolve
# dmaplane_trace.h during out-of-tree builds.
ccflags-y += -I$(src)

# Conditional GPU support — only build gpu_p2p.o if NVIDIA P2P headers present.
# When headers are not found, CONFIG_DMAPLANE_GPU is undefined and all GPU
# code is compiled out via #ifdef.  GPU ioctls return -ENODEV.
NV_HEADER := $(wildcard /usr/src/nvidia-*/nvidia/nv-p2p.h)
ifneq ($(NV_HEADER),)
  NV_DIR := $(dir $(firstword $(NV_HEADER)))
  dmaplane-objs += gpu_p2p.o
  ccflags-y += -DCONFIG_DMAPLANE_GPU -I$(NV_DIR)
endif

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
