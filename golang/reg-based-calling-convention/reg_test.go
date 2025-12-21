// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stackreg

import "testing"

func BenchmarkStack(b *testing.B) {
	stack(uint32(b.N))
}

func stack(n uint32)

func BenchmarkReg(b *testing.B) {
	reg(uint32(b.N))
}

func reg(n uint32)