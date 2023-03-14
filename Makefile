.PHONY: all clean

all:
	gcc -o matmul matmul.c -lOpenCL -lm -DCL_TARGET_OPENCL_VERSION=210

clean:
	rm -rf matmul
