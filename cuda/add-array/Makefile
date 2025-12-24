
all: cpp cu1 cu2 cu3

cpp: add.cpp
	g++ -O3 $^ -o hello
	time ./hello

cu1: add1.cu
	nvcc -O3 $^ -o hello
	time ./hello

cu2: add2.cu
	nvcc -O3 $^ -o hello
	time ./hello

cu3: add3.cu
	nvcc -O3 $^ -o hello
	time ./hello
