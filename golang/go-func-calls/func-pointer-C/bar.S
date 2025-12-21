
mytest:     file format elf64-littleaarch64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .text:

00000000004006e0 <bar>:
#include <stdio.h>

__attribute__((noinline, noipa))
int bar() {
  return 1;
}
  4006e0:	mov	w0, #0x1                   	// #1
  4006e4:	ret

Disassembly of section .fini:
