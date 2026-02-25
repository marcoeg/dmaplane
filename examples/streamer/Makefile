# examples/streamer/Makefile — Weight streaming TUI
#
# Prerequisites:
#   sudo apt install libncurses-dev
#   dmaplane.ko loaded, soft-RoCE configured
#
# Running (needs root for /dev/dmaplane access):
#   sudo ./weight_streamer_tui [ib_device]

CC      := gcc
CFLAGS  := -Wall -Wextra -O2 -std=gnu11
LDFLAGS :=

CFLAGS += -I../../include

PROGS := weight_streamer_tui

.PHONY: all clean

all: $(PROGS)

weight_streamer_tui: weight_streamer_tui.c ../../include/dmaplane_uapi.h
	$(CC) $(CFLAGS) -o $@ $< -lncurses $(LDFLAGS)

clean:
	rm -f $(PROGS)
