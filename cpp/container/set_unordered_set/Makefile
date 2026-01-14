all: test1 test2

test1: hello.cpp
	g++ -O3 $^ -o hello
	./hello

test2: sort.cpp
	g++ -O0 $^ -o hello
	./hello

clean:
	rm -rf ./hello
