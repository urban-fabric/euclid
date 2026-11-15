# euclid-matrix

**The matrix extensions beyond euclid's rank contract — the vocabulary, not the
algorithms.** Part of [euclid](../..).

These crates are not on crates.io. See [the root README](../../README.md#using-it)
for the git dependency to add.

euclid 0.2 deliberately shed linear-algebra *content* from the tensor contract:
`euclid-tensor` defines what a rank-n array **is** and stops there. This crate is
the other half — the factorization and solve vocabulary a backend implements.

It names no backend. A backend supplies the concrete tensor type these
monomorphize on — eagerly in `ramanujan`, compiled via XLA in `maxwell` — and
this crate never depends on either.

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

`MatrixTensor` (`Tensor + NormedDivisionAlgebra`) is the carrier bound these
signatures are written over; it exists to keep the `where` clauses readable, not
to add a requirement. `MatrixRealSpectrum` names the rank-1 eigenvalue/singular-value
tensor, which is real by construction even for a complex matrix.
`LazyDecompositions` is the blanket bundle of the five heavyweight traits, so a
portable kernel needing only the SPD spine can stay on `SpdSolve`.

## Not Here

No concrete tensor type, no backend, and no Hermitian-positive-definiteness
check — HPD-ness of an argument is the caller's precondition and implementors
need not verify it.

**No algorithms either, as of 0.4.** `identity`, `adjoint`, `gram`,
`least_squares`, `spd_inverse`, `inverse`, `quadratic_form` and
`dominant_eigenvector` used to live here, written once over the traits above.
They were deleted rather than kept, because euclid holds only what the fleet
must *agree* on: a Cholesky or a QR has many valid implementations, and choosing
between a blocked, packed or GPU path is exactly what a backend is for. What must
be agreed is the signature, which is what remains.

The cost is real and accepted — two backends wanting `least_squares` now write it
twice. The alternative was a shared algorithm that could not be tested without
euclid shipping a tensor, and two tested implementations beat one untested one.

## License

Apache-2.0, inherited from the workspace like every other member — see
[LICENSE](LICENSE) and [NOTICE](NOTICE).

This crate arrived from its own repository and carried its own `Proprietary`
licence until 0.4, on the reasoning that a move should not relicense code. That
has been overridden deliberately by the copyright holder: the fleet ships under
one licence. `cargo deny` is what surfaced it — `"Proprietary"` is not valid
SPDX, so the crate read as *unlicensed* rather than as restrictively licensed.
