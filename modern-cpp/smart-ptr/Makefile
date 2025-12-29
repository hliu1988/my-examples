all: test1 test2 test3

test1: unique.cpp
	g++ -std=c++20 $^ -o hello
	./hello

test2: shared.cpp
	g++ -std=c++20 $^ -o hello
	./hello

test3: weak.cpp
	g++ -std=c++20 $^ -o hello
	./hello
