package main

import "fmt"

func makeFunction(name string) func() {
	fmt.Println("00000")
	return makeFunction2(name)
}

func makeFunction2(name string) func() {
	fmt.Println("11111")
	return makeFunction3(name)
}

func makeFunction3(name string) func() {
	fmt.Println("33333")
	return func() {
		fmt.Printf(name)
	}
}

func main() {
	f := makeFunction("hellooo")
	f()
}
