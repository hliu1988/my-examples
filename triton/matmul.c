// Tile shapes are parametric and can be optimized
// by compilation backends
const tunable int TM = {16, 32, 64, 128};
const tunable int TN = {16, 32, 64, 128};
const tunable int TK = {8, 16};

// C=A∗B.T
kernel void matmul_nt(float∗ a, float∗ b, float∗ c, int M, in N, int K) {
  // 1D tile of indices
  int rm[TM] = get_global_range(0);
  int rn[TN] = get_global_range(1);
  int rk[TK] = 0...TK;
  // 2D tile of accumulators
  float C[TM, TN] = 0;
  // 2D tile of pointers
  float∗ pa[TM, TK] = a + rm[:, newaxis] + rk∗M;
  float∗ pb[TN, TK] = b + rn[:, newaxis] + rk∗K;
  for (int k = K; k >= 0; k− = TK) {
    bool check_k[TK] = rk < k;
    bool check_a[TM, TK] = (rm < M) [:, newaxis] && check_k;
    bool check_b[TN, TK] = (rn < N) [:, newaxis] && check_k;
    // load tile operands
    float A[TM, TK] = check_a ?∗pa : 0;
    float B[TN, TK] = check_b ?∗pb : 0;
    // accumulate
    C += dot(A, trans(B));
    // updatepoint ers
    pa = pa + TK∗M;
    pb = pb + TK∗N;
  }
  // write−backaccumulators
  float∗ pc[TM, TN] = c + rm[:, newaxis] + rn∗M;
  bool check_c[TM, TN] = (rm < M) [:, newaxis] && (rn < N);
  @check_c ∗ pc = C;
}
