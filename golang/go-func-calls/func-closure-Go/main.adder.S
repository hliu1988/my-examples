
mytest:     file format elf64-littleaarch64


Disassembly of section .text:

000000000009f680 <main.adder>:
package main

import "fmt"

//go:noinline
func adder() func(int) int {
   9f680:	ldr	x16, [x28, #16]
   9f684:	cmp	sp, x16
   9f688:	b.ls	9f6f0 <main.adder+0x70>  // b.plast
   9f68c:	str	x30, [sp, #-48]!
   9f690:	stur	x29, [sp, #-8]
   9f694:	sub	x29, sp, #0x8
	sum := 0
   9f698:	adrp	x0, a8000 <type:*+0x8000>
   9f69c:	add	x0, x0, #0x940
   9f6a0:	bl	1bb00 <runtime.newobject>   ; arg: type info of "sum"; ret: 0x400382040
   9f6a4:	str	x0, [sp, #32]
	return func(x int) int {
   9f6a8:	adrp	x0, af000 <type:*+0xf000>
   9f6ac:	add	x0, x0, #0x6a0
   9f6b0:	bl	1bb00 <runtime.newobject>   ; ret: 0x400384050, the new func object

   9f6b4:	adrp	x1, 9f000
   9f6b8:	add	x1, x1, #0x700
   9f6bc:	str	x1, [x0]
   9f6c0:	adrp	x27, 190000
   9f6c4:	ldr	w1, [x27, #2720]
   9f6c8:	cbnz	w1, 9f6d4 <main.adder+0x54>
   9f6cc:	ldr	x1, [sp, #32]               ; sum
   9f6d0:	b	9f6e0 <main.adder+0x60>
   9f6d4:	bl	7ce80 <runtime.gcWriteBarrier1>
   9f6d8:	ldr	x1, [sp, #32]
   9f6dc:	str	x1, [x25]

   9f6e0:	str	x1, [x0, #8]                ; write sum to *func-ptr
   9f6e4:	ldp	x29, x30, [sp, #-8]
   9f6e8:	add	sp, sp, #0x30
   9f6ec:	ret
func adder() func(int) int {
   9f6f0:	mov	x3, x30
   9f6f4:	bl	7ac30 <runtime.morestack_noctxt.abi0>
   9f6f8:	b	9f680 <main.adder>
   9f6fc:	udf	#0
