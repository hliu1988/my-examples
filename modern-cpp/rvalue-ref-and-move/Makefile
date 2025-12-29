all: test1 test2 test3

test1: rvalue-ref.cpp
	g++ -std=c++20 $^ -o hello
	./hello

test2: std-move.cpp
	g++ -std=c++20 $^ -o hello
	./hello

test3: move-ctor.cpp
	g++ -std=c++20 $^ -o hello
	./hello
