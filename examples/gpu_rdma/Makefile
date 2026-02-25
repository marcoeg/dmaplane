CC := gcc
SENDER_CFLAGS := -Wall -Wextra -O2 -I../../include
SENDER_LDFLAGS := -lcuda -lcudart -lm

RECEIVER_CFLAGS := -Wall -Wextra -O2 -I../../include
RECEIVER_LDFLAGS := -lm

# Check for CUDA — sender requires it, receiver does not.
HAVE_CUDA := $(shell echo 'int main(){}' | $(CC) -x c - -lcuda -lcudart -o /dev/null 2>/dev/null && echo yes)

TARGETS := gpu_receiver
ifeq ($(HAVE_CUDA),yes)
TARGETS += gpu_sender
endif

all: $(TARGETS)

gpu_sender: gpu_sender.c
	$(CC) $(SENDER_CFLAGS) -o $@ $< $(SENDER_LDFLAGS)

gpu_receiver: gpu_receiver.c
	$(CC) $(RECEIVER_CFLAGS) -o $@ $< $(RECEIVER_LDFLAGS)

clean:
	rm -f gpu_sender gpu_receiver

.PHONY: all clean
