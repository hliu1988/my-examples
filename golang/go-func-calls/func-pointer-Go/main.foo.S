
mytest:     file format elf64-littleaarch64


Disassembly of section .text:

0000000000077430 <main.foo>:

var gf func() int

//go:nosplit
//go:noinline
func foo(f func() int) (r int) {
   77430:	str	x30, [sp, #-32]!
   77434:	stur	x29, [sp, #-8]
   77438:	sub	x29, sp, #0x8
	r = f()
   7743c:	ldr	x1, [x0]         ; read *0x97e58 -> main.bar 0x77420
   77440:	mov	x26, x0
   77444:	blr	x1
   77448:	str	x0, [sp, #16]
	r += gf()
   7744c:	adrp	x27, 114000
   77450:	ldr	x26, [x27, #256] ; read *main.gf -> 0x97e58
   77454:	ldr	x1, [x26]        ; read *0x97e58 -> main.bar 0x77420
   77458:	blr	x1
   7745c:	ldr	x1, [sp, #16]
   77460:	add	x0, x1, x0
	return
   77464:	ldp	x29, x30, [sp, #-8]
   77468:	add	sp, sp, #0x20
   7746c:	ret
