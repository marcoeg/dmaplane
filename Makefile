.PHONY: all clean driver tests examples

all: driver tests examples

driver:
	$(MAKE) -C driver

tests:
	$(MAKE) -C tests

examples:
	$(MAKE) -C examples/misc
	-$(MAKE) -C examples/gpu_rdma 2>/dev/null || true
	-$(MAKE) -C examples/kvcache 2>/dev/null || true

clean:
	$(MAKE) -C driver clean
	$(MAKE) -C tests clean
	$(MAKE) -C examples/misc clean
	-$(MAKE) -C examples/gpu_rdma clean 2>/dev/null || true
	-$(MAKE) -C examples/kvcache clean 2>/dev/null || true
