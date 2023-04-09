.PHONY: all clean

all:
	gcc -o cpu cpu.c -lm
	gcc -o gpu gpu.c -lOpenCL -lm -DCL_TARGET_OPENCL_VERSION=210

clean:
	rm -rf gpu
	rm -rf cpu
	rm -rf output*
