package main

import "fmt"

type Value struct{}

func (Value) M(int) {}

type Pointer struct{}

func (*Pointer) M(int) {}

type MyInterface interface{ M(int) }

func describe(i MyInterface) {
	fmt.Printf("(%v, %T)\n", i, i)
}

func main() {
	// direct call of method with value receiver two spellings, but same
	var v Value

	// direct call of method with pointer receiver
	// two spellings, but same
	var p Pointer

	// indirect call of method on interface (×3)
	var i MyInterface

	i = v
	describe(i)

	i.M(1) // interface / containing value with value method
	i = &v
	describe(i)

	i.M(1) // interface / containing pointer with value method
	i = &p
	describe(i)

	i.M(1) // interface / containing pointer with pointer method

	MyInterface.M(i, 1)
	MyInterface.M(v, 1)
	MyInterface.M(&p, 1)
}
({}, main.Value)
(&{}, *main.Value)
(&{}, *main.Pointer)
