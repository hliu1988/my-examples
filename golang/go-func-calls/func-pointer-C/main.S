
mytest:     file format elf64-littleaarch64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .text:

0000000000400580 <main>:
  int r = (*f)();
  r += (*gf)();
  return r;
}

int main() {
  400580:	stp	x29, x30, [sp, #-16]!
  gf = &bar;
  400584:	adrp	x1, 420000
  400588:	adrp	x0, 400000
int main() {
  40058c:	mov	x29, sp
  gf = &bar;
  400590:	add	x0, x0, #0x6e0
  400594:	str	x0, [x1, #48]
  int (*lf)();
  lf = &bar;
  int r = foo(lf);
  400598:	bl	4006f0 <foo>
  printf("%d", r);
  40059c:	mov	w1, w0
  4005a0:	adrp	x0, 400000 <__abi_tag-0x278>
  4005a4:	add	x0, x0, #0x748
  4005a8:	bl	400540 <printf@plt>
}
  4005ac:	mov	w0, #0x0                   	// #0
  4005b0:	ldp	x29, x30, [sp], #16
  4005b4:	ret

Disassembly of section .fini:
