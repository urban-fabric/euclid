# Sketch: Vector / Matrix / Tensor contracts (XLA-aware)

Draft contracts, **now committed (no impls)** in `euclid::array` — `LazyVector`,
`LazyMatrix`, `LazyTensor`. The rank-graded peers of the scalar `LazyReal`, drawn so
they lower cleanly to XLA — including TPU — and so a single generic kernel runs on
the `f64`/`ramanujan` reference path **and** the `lebesgue`/tensor path.

## The four ranks (matching the `Scalar` naming)

```
rank 0   Scalar    LazyReal     elementwise analysis + scalar lift     euclid::numeric [exists]
rank 1   Vector    LazyVector   inner product, norms, vector space     euclid::array   [draft]
rank 2   Matrix    LazyMatrix   matmul + the SPD linear-algebra spine  euclid::array   [draft]
rank n   Tensor    LazyTensor   shape ops, reductions, dot_general     euclid::array   [draft]
```

Three deliberate decisions:

1. **`LazyTensor: LazyReal`; `LazyVector`/`LazyMatrix` are not.** A tensor *is*,
   elementwise, a lazy real — every transcendental/special
   (`exp`/`erf`/`ln_gamma`/`bessel_i0e`/…) applies elementwise for free, and the 170
   scalar kernels (`normal_log_pdf<T>`, `kl_gamma<T>`, …) become **batched** tensor
   kernels by monomorphizing at `T = Tensor`. A `LazyVector`/`LazyMatrix`, by
   contrast, carries only **vector-space + linear-algebra** structure — you don't
   `exp` a `DenseMatrix` through the same trait — so `ramanujan::DenseMatrix` (which
   is *not* elementwise-analytic) can implement it.

2. **`LazyMatrix` is parameterized over its scalar and its vector**
   (`type Scalar: LazyReal`, `type Vector: LazyVector<Scalar = Self::Scalar>`), so
   both `ramanujan::DenseMatrix` (`Scalar = f64`, `Vector = DenseVector`) and a rank-2
   `lebesgue::Tensor` (`Scalar = Vector = Self`) implement it without a cycle.

3. **A tensor *can also be a 1-D tensor*.** For the unified `lebesgue::Tensor`, one
   type satisfies `LazyTensor` at any rank **and** `LazyVector` (rank 1) and
   `LazyMatrix` (rank 2). For the typed `ramanujan` world, `DenseVector`/`DenseMatrix`
   are distinct types. The rank-graded traits support both shapes of backend.

## XLA / TPU tier legend

- **T1** native HLO op, hardware-lowered, TPU-first-class, fuses, differentiable.
- **T2** composed from T1, pure-HLO, stable, TPU-clean.
- **T3** custom-call (LAPACK/cuSOLVER) on CPU/GPU; pure-HLO Jacobi/blocked on TPU —
  works everywhere but slow/precision-sensitive. **Fence off the hot path.**
- **META** host-side metadata, not a device op.

---

## The contracts (now in `euclid::array`)

The trait definitions live in `euclid/src/array.rs` (committed, no impls). Rather than
duplicate them here, this is the **op → XLA HLO → tier** map they encode:

| Trait | Method | XLA HLO | Tier |
|-------|--------|---------|------|
| `LazyVector` | `dot` / `sum` / `norm_squared` | `DotGeneral` / `Reduce` | T1 |
| `LazyVector` | `norm` | `√(Reduce)` | T2 |
| `LazyVector` | `add` / `sub` / `scale` / `hadamard` | elementwise | T1 |
| `LazyMatrix` | `matmul` / `matvec` | `DotGeneral` | T1 |
| `LazyMatrix` | `transpose` | `Transpose` | T1 |
| `LazyMatrix` | `cholesky` | `Cholesky` | T1 |
| `LazyMatrix` | `triangular_solve` | `TriangularSolve` | T1 |
| `LazyMatrix` | `trace` / `diagonal` | `GetDiagonal` → `Reduce` | T2 |
| `LazyMatrix` | `logdet_spd` | `Cholesky` → `diag` → `ln` → `Reduce` | T2 |
| `LazyMatrix` | `solve_spd` | `Cholesky` + 2×`TriangularSolve` | T2 |
| `LazyTensor` | `reshape`/`transpose`/`broadcast_to`/`concat` | direct | T1 |
| `LazyTensor` | `reduce_sum`/`reduce_max`/`reduce_prod` | `Reduce` | T1 |
| `LazyTensor` | `dot_general` | `DotGeneral` | T1 |
| `LazyTensor` | `ge` → `Mask`, `select` | `Compare` / `Select` | T1 |

The whole core is **T1/T2** — native, stable, TPU-clean. There is deliberately **no
`inverse`**; the **T3** decompositions (`lu`/`qr`/`eigh`/`svd`/`determinant`) are a
*separate, fenced* `Decompositions` extension (not in the core traits), because on TPU
they are the slow pure-HLO Jacobi/blocked path.

What the SPD spine buys, branch-free and TPU-clean:
- **MVN log-density**: `δ = x − µ`; `L = Σ.cholesky()`; `z = L.triangular_solve(δ, …)`;
  `maha = z.norm_squared()`; `−½(maha + Σ.logdet_spd() + d·ln 2π)`.
- **MV Student-t, Wishart densities, GP conditioning, NIW/NW evidence** — all the same
  `cholesky`/`triangular_solve`/`logdet_spd`/`matmul` vocabulary.

The `Mask` + `select` story: `LazyReal` has no `bool`-returning compare on purpose; at
the tensor tier a compare yields a `PRED` tensor and control flow is `select`, never a
branch. That's the array form of branch-free — exactly what TPU's static-shape model
needs. Dynamic-shape ops (`nonzero`, boolean compaction, data-dependent slice) are
absent on purpose; they don't lower to TPU.

## How the existing surface lifts

- **Scalar kernels (`<T: LazyReal>`)** — run unchanged at `T = Tensor`, giving
  batched/elementwise XLA kernels. No rewrite. This is the bulk of the 170.
- **Vector kernels (`&[T]` folds)** — get tensor-native twins that use
  `reduce_sum`/`reduce_max` instead of a Rust fold (axis reductions, not loops). The
  `&[T]` forms stay for the `f64` reference and tiny cases.
- **Matrix kernels (new)** — written once on `LinearAlgebra`, run on `DenseMatrix`
  (f64 reference) and `Tensor` (XLA), the same way scalar kernels run on `f64` and
  `Tensor`.

## TPU reality to bake in (from the XLA discussion)

- **No fp64 on TPU** — f64 silently downcasts to fp32. The `1e-12` CPU parity is
  fp32 (~1e-5/1e-6) on device. SPD path needs jitter/whitening, not "use f64".
- **matmul defaults to bf16** on the MXU — `dot_general`/`matmul` for linalg must run
  at `HIGHEST` precision or Cholesky loses bits.
- **Static shapes, branch-free, index-free** — already our discipline; it *is* the
  TPU-lowerability checklist. `Mask`+`select` instead of `bool`+`if`; whole-array ops
  instead of scalar indexing.

## Resolved

- **Four rank-graded traits** (`Scalar`/`Vector`/`Matrix`/`Tensor`), `LazyMatrix`
  carrying `type Vector` so typed (`ramanujan`) and unified (`lebesgue::Tensor`,
  rank-1) backends both fit. A tensor *can also be a 1-D tensor*.

## Open questions

1. **Batch dims.** Does `LazyMatrix` carry leading batch axes in the contract, or stay
   strictly rank-2 and let `LazyTensor::dot_general` / a `vmap` provide batching? (A
   batch of covariances is the common case in GP / hierarchical models.)
2. **`Mask` placement + f64 reference.** `Mask` lives on `LazyTensor`; does the `f64`
   reference path need a degenerate scalar mask (`Mask = bool`) so scalar `select`
   type-checks, or do we keep `select` tensor-only?
3. **Precision policy.** Does the contract expose a precision knob (e.g. `matmul`
   precision for the TPU MXU), or is it entirely the backend's concern?
4. **Operators vs. methods.** `LazyVector`/`LazyMatrix` use `add`/`sub`/`scale`
   methods for explicitness; do we also want `Add`/`Sub` operator impls for ergonomics?
