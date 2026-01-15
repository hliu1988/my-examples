test: hello.cpp
	g++ -O0 --coverage $^ -o hello
	./hello
	mkdir -p gcov-report
	gcovr -r ./ . --html -o gcov-report/coverage.html --html-details

clean:
	rm -rf hello *.gcda *.gcno
