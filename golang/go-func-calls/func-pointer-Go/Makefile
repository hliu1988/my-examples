build:
	rm -rf go.mod
	go mod init mytest
	go build
	obj2cfg mytest main.foo
	obj2cfg mytest main.bar
	obj2cfg mytest main.main
