build:
	rm -rf go.mod
	go mod init mytest
	go build

dump: build
	obj2cfg mytest main.main
	obj2cfg mytest main.adder
	obj2cfg mytest main.adder.func1

clean:
	rm -rf mytest go.mod
