# euclid-matrix

**The matrix extensions beyond euclid's rank contract.** Part of [euclid](../..).

```toml
[dependencies]
euclid-matrix = "0.3"
euclid-tensor = "0.3"    # the Tensor contract these extend
euclid-numeric = "0.3"   # for the element trait bounds
```

euclid 0.2 deliberately shed linear-algebra *content* from the tensor contract:
`euclid-tensor` defines what a rank-n array **is** and stops there. This crate is
the other half — the factorization and solve vocabulary a backend implements, plus
the algorithms written once over that vocabulary.

It names no backend. A backend supplies the concrete tensor type these
monomorphize on — eagerly in `ramanujan`, compiled via XLA in `maxwell` — and
this crate never depends on either.

```rust,ignore
use euclid_matrix::least_squares;

// One source — runs on any Tensor backend, eager or compiled.
let x = least_squares(a, b);
```

## Rank, Not Separate Matrix and Vector Types

A vector is a rank-1 tensor and a matrix a rank-2 tensor; there are no separate
`Vector`/`Matrix` contracts. Three consequences shape this whole API:

- `SpdFactor` carries **one** associated tensor type, not a `Vector`/`Matrix`
  pair. The old `solve_vec`/`solve` and `whiten_vec`/`whiten` pairs collapsed
  into `solve` and `whiten` — rank travels with the value, so the two spellings
  had become the same signature.
- Element type is a **refinement bound on `Self`**, not an associated `Scalar`.
  A tensor *is* its numeric carrier, so `conjugate` is elementwise and
  `Determinant::determinant` returns a rank-0 `Self` rather than a distinct
  scalar type.
- Rank cannot be expressed in the bound, so it is documented per item. Every
  square-matrix argument is rank-2 and every vector argument rank-1 unless an
  item says otherwise.

## Backend Primitives

Traits a backend implements, because each maps to a hardware or library call
rather than being derivable from the rank contract:

| trait | provides |
|-------|----------|
| `SpdSolve` | `solve_spd` — the HPD solve `Σ⁻¹B`, a `POTRF`/`POTRS` pair eagerly, a compiled Cholesky-solve on XLA |
| `PreparedSpd` / `SpdFactor` | the **prepare once, solve many** seam: hoists the O(n³) factorization out of the per-call path, exposing `solve`, `whiten`, `logdet` |
| `TriangularSolve` | `solve_triangular` over `Triangle` and `TransposeMode`, with an optional unit diagonal |
| `GeneralSolve` | `solve` — the general nonsingular `A⁻¹B` |
| `Determinant` | `determinant`, as a rank-0 tensor |
| `QrDecomposition` | `qr` — reduced/economy `A = QR` |
| `HermitianEigendecomposition` | `eigh` — `A = V diag(λ) Vᴴ`, eigenvalues ascending |
| `SingularValueDecomposition` | `svd` — `A = U diag(σ) Vᴴ`, singular values descending |

`MatrixTensor` (`Tensor + NormedDivisionAlgebra`) is the carrier bound the
algorithms are generic over; it exists to keep the `where` clauses readable, not
to add a requirement. `MatrixRealSpectrum` names the rank-1 eigenvalue/singular-value
tensor, which is real by construction even for a complex matrix.
`LazyDecompositions` is the blanket bundle of the five heavyweight traits, so a
portable kernel needing only the SPD spine can stay on `SpdSolve`.

## Algorithms

Written once over the traits above — a backend gets these for free:

| function | notes |
|----------|-------|
| `identity` | derived from the rank contract by one-hot-encoding `iota`, not asked of the backend, so every backend is correct by construction |
| `adjoint` | `Aᴴ` — elementwise conjugation composed with an axis swap; the plain transpose for a real carrier |
| `gram` | `AᴴA`. Forming it squares the condition number; prefer a QR path when available |
| `least_squares` | `A·x ≈ B` via the normal equations through the SPD path — same condition-number caveat |
| `spd_inverse` / `inverse` | `Σ \ I` and `A \ I`. Prefer *applying* an inverse via the solve traits to forming one |
| `quadratic_form` | `xᴴAx`, real for Hermitian `A`. The real part is taken, not asserted — that is the documented contract |
| `dominant_eigenvector` | power iteration for a **fixed** step count: a compiled backend cannot branch on a device value without a host sync this contract has no room for. Callers needing a tolerance iterate themselves |

## Not Here

No concrete tensor type, no backend, and no Hermitian-positive-definiteness
check — HPD-ness of an argument is the caller's precondition and implementors
need not verify it.

## License

euclid-matrix keeps its own `Proprietary` licence rather than inheriting the
workspace's Apache-2.0. It arrived here from its own repository, and a move
should not relicense code.
