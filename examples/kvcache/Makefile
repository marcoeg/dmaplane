CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=gnu11 -I../../include
LDFLAGS = -lm

# CUDA detection — check common header locations
CUDA_FOUND :=
ifneq ($(wildcard /usr/local/cuda/include/cuda_runtime.h),)
CUDA_FOUND := 1
CUDA_CFLAGS  = -I/usr/local/cuda/include
CUDA_LDFLAGS = -L/usr/local/cuda/lib64 -lcudart
else ifneq ($(wildcard /usr/include/cuda_runtime.h),)
CUDA_FOUND := 1
CUDA_CFLAGS  =
CUDA_LDFLAGS = -lcudart
endif

TARGETS := kvcache_sender kvcache_receiver

all: $(TARGETS)

kvcache_sender: kvcache_sender.c kvcache_common.h
ifdef CUDA_FOUND
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -DHAVE_CUDA -o $@ $< $(LDFLAGS) $(CUDA_LDFLAGS)
else
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
endif

kvcache_receiver: kvcache_receiver.c kvcache_common.h
ifdef CUDA_FOUND
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -DHAVE_CUDA -o $@ $< $(LDFLAGS) $(CUDA_LDFLAGS)
else
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
endif

clean:
	rm -f $(TARGETS)

.PHONY: all clean
