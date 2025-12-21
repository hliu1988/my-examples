package main

var sum int64

//go:noinline
//go:nosplit
func addUpDirect(s []int64) {
	for i := 0; i < len(s); i++ {
		sum += s[i]
	}
}

//go:noinline
//go:nosplit
func addUpViaInterface(s []interface{}) {
	for i := 0; i < len(s); i++ {
		sum += s[i].(int64)
	}
}

func main() {
	is := []int64{0x55, 0x22, 0xab, 0x9}

	addUpDirect(is)

	iis := make([]interface{}, len(is))
	for i := 0; i < len(is); i++ {
		iis[i] = is[i]
	}

	addUpViaInterface(iis)
}
