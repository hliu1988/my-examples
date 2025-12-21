
mytest:     file format elf64-littleaarch64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .text:

00000000004006f0 <foo>:
}

int (*gf)();

__attribute__((noinline, noipa))
int foo(int (*f)()) {
  4006f0:	stp	x29, x30, [sp, #-32]!
  4006f4:	mov	x29, sp
  4006f8:	str	x19, [sp, #16]
  int r = (*f)();
  4006fc:	blr	x0
  400700:	mov	w19, w0
  r += (*gf)();
  400704:	adrp	x1, 420000 <__libc_start_main@GLIBC_2.34>
  400708:	ldr	x0, [x1, #48]
  40070c:	blr	x0
  return r;
}
  400710:	add	w0, w19, w0
  400714:	ldr	x19, [sp, #16]
  400718:	ldp	x29, x30, [sp], #32
  40071c:	ret

Disassembly of section .fini:
