
mytest:     file format elf64-littleaarch64


Disassembly of section .text:

000000000009f700 <main.adder.func1>:
import "fmt"

//go:noinline
func adder() func(int) int {
	sum := 0
	return func(x int) int {
   9f700:	ldr	x1, [x26, #8]   ; x26: 0x4000384050; load sum-ptr
		sum += x
   9f704:	ldr	x2, [x1]        ; load sum: 0
   9f708:	add	x0, x2, x0
   9f70c:	str	x0, [x1]
		return sum
   9f710:	ret
	...
