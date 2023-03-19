.PHONY: all clean

all:
	gcc -o matmul matmul.c -lOpenCL -lm -DCL_TARGET_OPENCL_VERSION=210
	gcc -o matmul_sw matmul_sw.c

clean:
	rm -rf matmul
	rm -rf matmul_sw
