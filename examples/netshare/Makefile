# examples/netshare/Makefile — Cross-machine RDMA host path
#
# Prerequisites:
#   sudo apt install libibverbs-dev librdmacm-dev rdma-core
#
# GPU path (gpu_sender/gpu_receiver) lives in examples/gpu_rdma/

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I../../include
LDFLAGS = -libverbs -lrdmacm -lm

all: sender receiver

sender: sender.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

receiver: receiver.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f sender receiver

.PHONY: all clean
