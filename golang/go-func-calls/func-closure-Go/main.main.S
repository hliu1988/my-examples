
mytest:     file format elf64-littleaarch64


Disassembly of section .text:

000000000009f720 <main.main>:
		sum += x
		return sum
	}
}

func main() {
   9f720:	ldr	x16, [x28, #16]
   9f724:	cmp	sp, x16
   9f728:	b.ls	9f808 <main.main+0xe8>  // b.plast
   9f72c:	str	x30, [sp, #-128]!
   9f730:	stur	x29, [sp, #-8]
   9f734:	sub	x29, sp, #0x8
	pos, neg := adder(), adder()
   9f738:	bl	9f680 <main.adder>
   9f73c:	str	x0, [sp, #72]
   9f740:	bl	9f680 <main.adder>
   9f744:	str	x0, [sp, #80]
   9f748:	mov	x1, xzr
	for i := 0; i < 10; i++ {
   9f74c:	b	9f7f4 <main.main+0xd4>
   9f750:	str	x1, [sp, #48]
		fmt.Println(
			pos(i),
   9f754:	ldr	x26, [sp, #72]  ; 0x4000384050, the new func object (func-ptr)
   9f758:	ldr	x2, [x26]       ; 0x9f700, aka main.adder.func1
   9f75c:	mov	x0, x1          ; 1st arg: 0
   9f760:	blr	x2
   9f764:	str	x0, [sp, #64]
			neg(-2*i),
   9f768:	ldr	x26, [sp, #80]
   9f76c:	ldr	x1, [x26]
   9f770:	ldr	x2, [sp, #48]
   9f774:	neg	x3, x2, lsl #1
   9f778:	mov	x0, x3
   9f77c:	blr	x1
   9f780:	str	x0, [sp, #56]
		fmt.Println(
   9f784:	stp	xzr, xzr, [sp, #88]
   9f788:	stp	xzr, xzr, [sp, #104]
			pos(i),
   9f78c:	ldr	x0, [sp, #64]
   9f790:	bl	71d00 <runtime.convT64>
   9f794:	adrp	x1, a8000 <type:*+0x8000>
   9f798:	add	x1, x1, #0x940
   9f79c:	str	x1, [sp, #88]
   9f7a0:	str	x0, [sp, #96]
			neg(-2*i),
   9f7a4:	ldr	x0, [sp, #56]
   9f7a8:	bl	71d00 <runtime.convT64>
   9f7ac:	adrp	x1, a8000 <type:*+0x8000>
   9f7b0:	add	x1, x1, #0x940
   9f7b4:	str	x1, [sp, #104]
   9f7b8:	str	x0, [sp, #112]

// Println formats using the default formats for its operands and writes to standard output.
// Spaces are always added between operands and a newline is appended.
// It returns the number of bytes written and any write error encountered.
func Println(a ...any) (n int, err error) {
	return Fprintln(os.Stdout, a...)
   9f7bc:	adrp	x27, 16a000 <runtime.itabTableInit+0xba0>
   9f7c0:	ldr	x2, [x27, #1160]
	for i := 0; i < 10; i++ {
   9f7c4:	ldr	x3, [sp, #48]
   9f7c8:	add	x3, x3, #0x1
   9f7cc:	str	x3, [sp, #48]
   9f7d0:	adrp	x0, e4000 <runtime.gcbss+0x20>
   9f7d4:	add	x0, x0, #0xf38
   9f7d8:	orr	x4, xzr, #0x2
   9f7dc:	mov	x1, x2
   9f7e0:	add	x2, sp, #0x58
   9f7e4:	mov	x3, x4
   9f7e8:	bl	9a9f0 <fmt.Fprintln>
			neg(-2*i),
   9f7ec:	ldr	x0, [sp, #80]
	for i := 0; i < 10; i++ {
   9f7f0:	ldr	x1, [sp, #48]
   9f7f4:	cmp	x1, #0xa
   9f7f8:	b.lt	9f750 <main.main+0x30>  // b.tstop
		)
	}
}
   9f7fc:	ldp	x29, x30, [sp, #-8]
   9f800:	add	sp, sp, #0x80
   9f804:	ret
func main() {
   9f808:	mov	x3, x30
   9f80c:	bl	7ac30 <runtime.morestack_noctxt.abi0>
   9f810:	b	9f720 <main.main>
	...
