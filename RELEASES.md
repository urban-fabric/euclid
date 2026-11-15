# Releases

**Breaking changes only.** This is not a changelog — it does not list what was added, fixed or
made faster, because a consumer reading it has exactly one question: *what do I have to change to
move from the version I am pinned to?* Everything else is in `jj log`, where the squash-merged
PR body is the record.

See [CONTRIBUTING](CONTRIBUTING.md#releasing) for how a release is cut, and
[the README](README.md#using-it) for how to pin one.

## What counts as breaking

euclid is `0.x`, and the **minor number is the breaking-change axis**: `0.5.0 → 0.6.0` may break
you, `0.5.0 → 0.5.1` may not. That is cargo's own reading of a `0.x` version, so a requirement of
`"0.5"` accepts patches and refuses the next minor, which is the behaviour you want.

A change is breaking if a consumer that compiled against the previous version might not compile,
or might compute a different answer. For a repository that is mostly trait surface, that is wider
than it sounds and the second half is the part worth watching:

- a trait, method, associated type or constant renamed, removed, or given a new signature
- a **new required method** on an existing trait, which breaks every implementor
- a crate renamed or split
- a conformance **law** changed — a backend that passed may now fail, and a law is public API
- a documented convention changed, even with no signature change, because a formula written
  against the old reading now produces a different number

A relicence is listed too. It changes nothing at compile time and everything about what a
recipient may do.

## 0.5.0

**`euclid-property` is now `euclid-stability`.** Rename the dependency and the import
(`euclid_property` → `euclid_stability`), and the environment variables
(`EUCLID_PROPERTY_*` → `EUCLID_STABILITY_*`). The crate is `publish = false` and taken under
`[dev-dependencies]`, so this touches test code only.

The name was wrong rather than merely unfashionable: "property testing" names the type-driven
QuickCheck/proptest tradition, and this crate has no `Arbitrary` and never did — its weight is
numerical (condition numbers, forward-error bounds, seam hunting, accuracy budgets).

**Three crates changed licence, from `Proprietary` to Apache-2.0**: `euclid-algebra`,
`euclid-scalar` and `euclid-matrix`. Every member now inherits the workspace licence, so the
whole repository is Apache-2.0 with no exceptions. If you were relying on those three being
proprietary, they are not any more — and note `"Proprietary"` was never valid SPDX, so tooling
read them as *unlicensed* rather than as restrictively licensed.

**`euclid_tensor::conformance::structural_laws` corrected its 3-cycle at rank ≥ 4.** If your
tensor is rank-4 or higher and you run the published laws, this changes your result — for the
better: the old law built `[1, 2, 0, 1, 3]` for a rank-4 tensor, which is five entries with axis
1 twice and not a permutation at all. It was correct only when rank was exactly 3, which is where
euclid's own reference sample stopped. **A backend that was failing this law was probably not
wrong.** Re-run it.

**`MatrixRealSpectrum` gained a required method, `from_spectrum`.** Every implementor must add
it — a one-line embedding of a real spectrum into the carrier (the identity for a real carrier,
`λ + 0i` for a complex one).

It is there because without it the defining property of an eigendecomposition could not be
*stated*: `A = V·diag(λ)·Vᴴ` needs a `RealSpectrum` and a `Self` in one expression, and nothing
related the two types. `HermitianEigendecomposition` and `SingularValueDecomposition` were
consequently the only part of euclid's contract surface with no law behind them, while the
`conformance` module's header advertised laws for both. Both laws now exist
(`conformance::eigh_laws`, `conformance::svd_laws`).

**`svd` returns `V`, not `Vᴴ`** — now stated on the trait, where it was previously only implied
by the formula. LAPACK's `gesdd` hands back `Vᵗ`, so a backend wrapping it directly must
transpose before returning. Getting this backwards yields a correctly-shaped and silently wrong
result for any square input, which is why it is worth a paragraph rather than a formula. `eigh`
likewise returns `(λ, V)` with eigenvectors as *columns*.

**The generated conformance strategy is now exported, and enabling `conformance` pulls
`euclid-stability`.** Additive — nothing that compiled stops compiling — but it changes what the
feature costs, so it is listed.

Each contract crate's `conformance` module gained a `generated` submodule holding the search
euclid runs against its own carriers: `tensor_laws_generated`, `key_laws_generated`,
`quantity_laws_generated`, `semiring_laws_generated`, `real_laws_generated` and their siblings.
`euclid-tensor` and `euclid-units` build your carrier through `Tensor::from_host` and
`Measure::from_raw`, so they need nothing from you; `euclid-algebra` and `euclid-numeric` need a
`GeneratedSample` impl, since their contracts contain no constructor.

`euclid-stability` is therefore an **optional runtime dependency** of the five contract crates
rather than purely dev-only. Default builds are unaffected and the dependency is absent from
them — verified with `cargo tree --edges normal`.

**`Tensor::transpose_axes` now documents a precondition it always had**: `axes` must be a
permutation of `0..rank()`. A shorter list is not a partial permutation leaving the remaining axes
alone; it is out of contract. No signature change, but if you were passing a short list, the
behaviour you were getting was never promised.

Within `euclid-stability` (dev-only, so test code only): `persist::load` returns `Vec<Entry>`
rather than `Vec<Vec<u64>>` — use `load_tapes` for the old shape; `persist::save` takes a
rendering argument; `shrink::minimise` returns a `Cost` rather than a `usize` probe count; and
`check` now requires `T: Debug`, which the `property!` macro satisfies for you.

## 0.4.0

**`oresme` is now `euclid-quantity`.** The heaviest rename in euclid's history — roughly 550 call
sites across the fleet — which is why it was done on its own.

**`euclid-scalar` was split out of `euclid-algebra`.** The concrete `Copy` number line
(`Real`/`Complex` over `f64`/`f32`) moved to its own crate. If you were importing
`euclid_algebra::LazyReal` or `LazyComplex`, those aliases are gone: the contract is
`euclid_numeric::Real` and the concrete number is `euclid_scalar::Real`, told apart by path
rather than by an import rename.

**`euclid-matrix` deleted its algorithms**, keeping only the trait signatures: `identity`,
`adjoint`, `gram`, `least_squares`, `spd_inverse`, `inverse`, `quadratic_form` and
`dominant_eigenvector` are gone. They were matrix *kernels*, and euclid holds only what the fleet
must agree on — a Cholesky or a QR has many valid implementations and choosing between blocked,
packed or GPU is a backend's job. An algorithm over those traits now lives with a backend, so two
backends wanting the same one write it twice. That is the accepted cost.

## 0.3.0

**The `euclid` facade crate was removed.** There is no umbrella; depend on the specific crates
you use. A manifest naming `euclid` does not resolve against 0.3 or later at all, which is the
failure mode that had `euclid-algebra` and `euclid-matrix` stale in their own repositories before
they were folded in.

## 0.2.0

**`LazyMatrix` and `LazyVector` were replaced by a single rank-n `Tensor`.** A vector is a rank-1
tensor and a matrix a rank-2 tensor; there are no separate vector and matrix contracts. Paired
methods collapsed accordingly — `solve_vec`/`solve` and `whiten_vec`/`whiten` became `solve` and
`whiten`, because rank travels with the value and the two spellings had become one signature.

Linear-algebra *content* left the tensor contract in the same release.

## Before 0.2.0

Not recorded. euclid was a single crate under heavy reshaping and nothing outside it was pinned,
so there is no migration to describe — the four-crate workspace split, the Apache-2.0 relicence
of the original crate, and the `numeric`/`algebra` collapse into the two-tier scalar foundation
all predate anything that could have been depending on them.
