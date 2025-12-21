
mytest:     file format elf64-littleaarch64


Disassembly of section .text:

00000000000732c0 <main.addUpDirect>:

var sum int64

//go:noinline
//go:nosplit
func addUpDirect(s []int64) {
   732c0:	str	x0, [sp, #8]
	for i := 0; i < len(s); i++ {
   732c4:	mov	x2, xzr
   732c8:	b	732e8 <main.addUpDirect+0x28>
		sum += s[i]
   732cc:	adrp	x27, 12a000 <runtime.mheap_+0x18c20>
   732d0:	ldr	x3, [x27, #416]
   732d4:	ldr	x4, [x0, x2, lsl #3]
   732d8:	add	x3, x3, x4
   732dc:	adrp	x27, 12a000 <runtime.mheap_+0x18c20>
   732e0:	str	x3, [x27, #416]
	for i := 0; i < len(s); i++ {
   732e4:	add	x2, x2, #0x1
   732e8:	cmp	x1, x2
   732ec:	b.gt	732cc <main.addUpDirect+0xc>
	}
}
   732f0:	ret
	...
