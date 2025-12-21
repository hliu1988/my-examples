build:
	rm -rf go.mod
	go mod init test
	go build

run: build
	./test

clean:
	rm -rf test *.S go.mod