all: main tiling

main:
	nvcc main.cu -o hello -lm
	./hello

no_shared:
	nvcc gemm-no-shared.cu -o hello -lm
	./hello

shared:
	nvcc gemm-shared.cu -o hello -lm
	./hello

clean:
	rm -rf ./hello
