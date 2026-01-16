
all: gcc llvm

gcc: hello.c
	gcc -O3 -S hello.c -o hello.gcc.s

llvm: hello.c
	clang -O3 -S hello.c -o hello.llvm.s
