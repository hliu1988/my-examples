package main

func bar() int {
	return 1
}

var gf func() int

//go:nosplit
//go:noinline
func foo(f func() int) (r int) {
	r = f()
	r += gf()
	return
}

//go:nosplit
func main() {
	gf = bar
	lf := bar
	r := foo(lf)
	println(r)
}
