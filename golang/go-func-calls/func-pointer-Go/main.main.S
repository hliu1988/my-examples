
mytest:     file format elf64-littleaarch64


Disassembly of section .text:

0000000000077470 <main.main>:
	r += gf()
	return
}

//go:nosplit
func main() {
   77470:	str	x30, [sp, #-32]!
   77474:	stur	x29, [sp, #-8]
   77478:	sub	x29, sp, #0x8
	gf = bar

   7747c:	adrp	x27, 13a000
   77480:	ldr	w1, [x27, #1104]
   77484:	cbz	w1, 77498
   77488:	adrp	x27, 114000
   7748c:	ldr	x1, [x27, #256]
   77490:	bl	74c30 <runtime.gcWriteBarrier1>
   77494:	str	x1, [x25]

   77498:	adrp	x0, 97000
   7749c:	add	x0, x0, #0xe58    ; 0x97e58 Q1? why not use main.bar (0x77420); 0x97ba8: go:func.*

   774a0:	adrp	x27, 114000
   774a4:	str	x0, [x27, #256]   ; 0x114100 -> main.gf
	lf := bar
	r := foo(lf)
   774a8:	bl	77430 <main.foo>
   774ac:	str	x0, [sp, #16]
	println(r)
   774b0:	bl	40340 <runtime.printlock>
   774b4:	ldr	x0, [sp, #16]
   774b8:	bl	409d0 <runtime.printint>
   774bc:	bl	40570 <runtime.printnl>
   774c0:	bl	403c0 <runtime.printunlock>
}
   774c4:	ldp	x29, x30, [sp, #-8]
   774c8:	add	sp, sp, #0x20
   774cc:	ret
