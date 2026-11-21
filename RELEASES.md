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

## 0.7.0

**`euclid-quantity` no longer re-exports `euclid-units`.** This is the one change here that can
stop you compiling, and the fix is mechanical: depend on `euclid-units` and import from it.

About sixty names left `euclid_quantity`'s surface — the five contracts (`Measure`, `Quantity`,
`ScalarQuantity`, `LinearQuantity`, `DiscreteQuantity`), `Dimension`, `dim`, and the whole
unit-marker set from `Length` and `Mass` through `Torque` and `VolumetricFlowRate`. They were
never defined there. A caller naming one now depends on the crate that defines and documents it,
so each name has exactly one path.

```toml
# add, if you were reaching these through euclid-quantity
euclid-units = { version = "0.7", path = "../euclid/crates/euclid-units" }
```

```rust
// before
use euclid_quantity::{Length, Mass, Quantity};
// after
use euclid_units::{Length, Mass, Quantity};
```

`HostQuantity`, `QuantityError` and `scalar_mul` originate in `euclid-quantity` and are
unaffected. Nothing else was removed from any crate's surface in this release: the conformance
and law modules were split into sub-topic files, but every one re-exports at its former path, so
`conformance::real_laws` and the rest resolve exactly as before.

**Many kernels return different — more accurate — answers.** Listed because this file counts a
changed answer as breaking, not because any of these is a regression. If you have reference
values recorded against 0.6.0, re-record them. The direction is always toward the correctly
rounded result, and every figure below is measured against an independent high-precision
reference rather than against a previous version of the kernel.

- **`bessel_j0`, `bessel_j1`, `bessel_y0`, `bessel_y1`** — the asymptotic phase was formed as
  `x − c`, whose rounding *is* the angle's absolute error. Up to `4.1e-10` absolute, against a
  documented promise of machine precision in absolute terms at every argument.
- **`erf_inv` and `erfc_inv`** — now correctly rounded. The far tail measured `204` ulp through
  `4.6e14` ulp across its bands, and there was a separate real defect near `|x| = 1`.
- **`ln_rising_factorial`** — `6.5e13` ulp, from differencing two log-gammas. Its threshold had
  been placed on *absolute* error while the answer grows like `n·ln a`.
- **`gamma`** above `x = 20` — Lanczos *truncation*, not arithmetic, reached `458` ulp at the top
  of the range; now `≤ 3.6` ulp throughout.
- **`trigamma`** — `53` ulp at `x = 20` where the problem itself is benign.
- **`erfcx` and `erfc`** — `23` ulp by `x = 1.36`; `erfcx` has its own Chebyshev route.
- **`bessel_i0`, `bessel_i1`** and their scaled forms — one to four ulp uniformly from `1` to the
  overflow edge. The `f32` Stirling remainder alone had cost about nine hundred ulp.
- **`gamma_lr`, `gamma_ur`** — the prefactor no longer forms a cancelling sum under `exp`; worst
  280 → 11 ulp, mean 38.9 → 2.0 on the measured grid.
- **`beta_reg`** — the prefactor is centred on the mode; median 10 → 4 ulp, mean 24 → 16.6.
- **`zeta`** — route boundaries moved to where the measurements put them, with a new third arm on
  `(−1/8, 0)`.
- **`ln_gamma`, `rgamma`, `polygamma`, `digamma`, `ln_beta`, `exp_int_e1`, `bessel_k0`,
  `bessel_k1`** — all have reformulated arms or moved crossovers. `bessel_k0`'s series/quadrature
  join moved from `2.0` to `1.25`, which costs about 9× on `(1.25, 2.0]` and buys 48 → 4 ulp.
- **The whole `f32` surface** was measured for the first time, and several families are now
  native single precision where they previously narrowed an `f64` result.

**`euclid-stability` gains `relation::forward_stable_sum`.** Additive, so it breaks nothing, but
worth knowing it exists: it bounds a sum whose reference itself cancels, and its failure message
reports the `relative / admissible` ratio you act on. Error budgets across the suites are now set
from each problem's conditioning rather than from picked tolerances, so a property that fails
tells you whether the kernel is unstable or the input is simply hard.

## 0.6.0

**`euclid-numeric` is merged into `euclid-scalar`.** Replace the dependency and import paths
(`euclid_numeric` → `euclid_scalar`). All numeric trait names, primitive implementations,
constants, interval helpers, compensated reductions and conformance laws now live in scalar.
The numeric crate is removed rather than retained as a compatibility facade.

The old eager `euclid_scalar::Real` and `Complex` traits are renamed to **`HostReal` and
`HostComplex`**. Existing consumers of that crate must rename those bounds and qualified
calls. The names `Real` and `Complex` now identify the backend-neutral traits formerly in
numeric: they permit `Clone` carriers and mask-valued comparisons. Use `HostReal` or
`HostComplex` when `Copy`, host predicates and associated constants are required.

The workspace, generated laws and test tooling version together at 0.6.0.

**`euclid-stability` assumptions now discard cases.** `assume!(false)` previously returned
success, so a property could meet its case count without exercising any assertion. It now
returns `FailureKind::Discard`; rejected cases contribute neither coverage nor target scores
and are replaced within `max_rejects`. All-rejected properties fail. Callers handling `Outcome`
directly should distinguish discards from failed assertions.

`Failure` gains `kind` and `identity` fields. Replace two-field struct literals with
`Failure::new(message, location)`; use `Failure::with_identity` for assertions sharing a helper.
`Config` gains `exploration` and `replay_checks`; add `..Config::default()` to full literals.
Failure shrinking now preserves the original assertion or panic location and rejects unstable
replays, so tests that relied on changing failure sites or stateful replays need adjustment.

Full-width signed integer and index generators now cover their entire declared domains, and
float interpolation handles ranges whose intermediate width overflows. Recheck persisted
tapes and accuracy budgets that depended on those formerly incorrect draws. Non-finite
`target()` calls now preserve any earlier finite score, as their documented ignore policy requires.

**The `conformance` feature no longer includes the generated search.** It is split: the
fixed-sample laws stay in `conformance`, and the `conformance::generated` module — the
`*_laws_generated` family and `GeneratedSample` — moves behind a new `conformance-generated`
feature. Five crates are affected: `euclid-algebra`, `euclid-scalar`, `euclid-units`,
`euclid-random`, `euclid-tensor`. (`euclid-matrix` has no generated module and is unchanged.)

If you enable `conformance` and call anything under `conformance::generated`, add
`conformance-generated` to your feature list; it enables `conformance` for you, so replacing one
with the other is enough. If you only call the slice- or sample-taking laws, nothing changes.

What it buys: `conformance` now resolves and compiles **no property engine at all**.
`cargo tree --edges normal --features conformance` shows zero `euclid-stability` edges where it
previously showed one, so a consumer checking a carrier against the contract no longer pays for
the search machinery. Note this is dependency hygiene and *not* a publishing fix — `cargo publish`
on these crates still fails while `euclid-stability` is `publish = false`, because cargo resolves
optional and versioned dev dependencies at publish time whether or not their feature is on.

**`Real::beta_reg_inv` at `f32` resolves a small root instead of returning the bracket
midpoint.** Listed because it changes computed answers. The `f32` bisection stopped once its
bracket fell below a bare `4·f32::EPSILON` — an *absolute* bound of `4.8e-7` — so for any root
below that it returned the midpoint of the remaining bracket rather than the root.
`beta_reg_inv(0.05, 0.1, 0.1)` came back as `2.38e-7`, which is `2⁻²²` and visibly the bisection
halting, against a true root of `8.66e-11`: **2,752 times too large**. Worst relative error
against the `f64` kernel over U-shaped shapes drops from `2.8e3` to `3.0e-7`, about 2.5 `f32`
ulp.

The bound is now relative to the bracket, matching `f32`'s own `gamma_lr_inv` and the `f64`
peer — this was the one of the three that did not scale. **Roots above `~5e-7` are unaffected**,
so a consumer whose shapes put the root mid-interval sees no change; the shapes that move are
U-shaped betas (either parameter below 1) and strongly asymmetric ones, where the mass piles
into a corner. `f64` was never affected.

**`Real::beta_reg` returns a probability at large shapes where it returned `NaN`.** Listed
because it changes computed answers, which this file counts as breaking even though every
changed answer was previously unusable. For `a + b` above roughly `5.6e102` the regularized
incomplete beta returned `NaN` for *every* `x`; it now returns the correct value — `I_½(a, a)`
is exactly `½`, by symmetry of `Beta(a, a)`, at every representable shape including `f64::MAX`.
Two narrower cases went the same way: `a·b` overflowing past `1.8e308`, and a shape ratio
extreme enough to underflow the variance while the standard deviation itself is representable
(`a = f64::MAX, b = 1e7`).

The Edgeworth branch scales the shapes when their sum would overflow and uses a compensated
residual to locate the input relative to the mean. For example, at `x = 0.5` with shapes
`f64::MAX` and `f64::MAX.next_down()`, the CDF returns 0 in that order and 1 in the reverse
order. The independent 60-digit reference grid measured a worst absolute error of `6.05e-9`
near the branch entry at `min(a, b) = 1e7`. Highly asymmetric shapes also use this
calculation when the shifted gamma limit would itself use Edgeworth.

**Empty tensor shapes have zero elements even when an earlier partial product overflows.**
`TensorSpec::num_elements()` returns `Some(0)` if any axis length is zero. Shapes with
nonzero extents still report an overflowing product as `None`.

**Gamma and beta boundary results changed.** The incomplete gamma and beta CDFs and their
quantiles require positive, finite shapes. Invalid shapes return `NaN`, including at support
endpoints. Gamma tails at large finite inputs can now saturate to `0` or `1` where an overflowing
intermediate returned `NaN`. At shapes so large that all interior gamma quantiles round to the
shape, the inverse returns that shape. The beta CDF uses the integrated density when either
shape is exactly one, and its large-shape skewness correction stays finite at large asymmetry.
Results in those beta regimes can change.

**Digamma near negative poles uses reflection.** Negative integer poles and negative infinity
return `NaN`. Nearby finite arguments can change because the calculation now preserves the
distance to the pole and avoids walking the recurrence across the entire negative axis.
The `f32` kernel now evaluates in `f64` and rounds once. This also fixes a wrong sign
at the last binary32 input below the positive zero of digamma.
Trigamma now reduces by the nearest integer before evaluating the reflection term,
which preserves the distance to remote poles in both precisions.

**Negative `f64` integer powers preserve representable subnormals.** When the positive
power overflows, Euclid raises the reciprocal of the base to the exponent magnitude.
For example, `Real::powi(2.0_f64, -1024)` now returns `2⁻¹⁰²⁴`.

**`Real::erfc_inv` at `f32` evaluates the inverse in `f64` and rounds once.** The smallest
positive `f32` inputs now give finite roots. Ordinary quantiles can change by a few ulps.

**Gamma and beta preserve more extreme tails.** The gamma quantile uses the upper tail for
probabilities above one-half. The small-shape gamma series computes the complement directly,
including representable subnormal probabilities. Small-shape `f32` quantiles round after
inversion. The beta CDF uses a shifted gamma limit for strongly asymmetric shapes and the
endpoint mass ratio for two tiny shapes. The log beta
calculation combines the large terms before rounding. The `f64` inverse complementary error
function solves in log probability where halving its argument would round a subnormal.

**Fixed-width rational arithmetic checks the reduced result.** `Rational64` and `Rational128`
reduce with wider intermediates and cancel common factors before multiplying. Operators panic
in both debug and release when the reduced result cannot be represented. To handle overflow
and zero divisors, use `checked_new` or the corresponding `checked_*` method. Implementations
of `euclid_algebra::Integer` must provide `checked_product`, `checked_negation`, and
`checked_fraction_sum`. The supplied `i64` and `i128` implementations provide those methods.

**Matrix conformance uses a scaled Frobenius bound.** The residual assertion handles finite
values whose squares over- or underflow and rejects invalid tolerances and non-finite data.

**Point and interval midpoint calculations use the primitive midpoint.** Large same-sign
interval bounds can now have a finite midpoint, and opposite-sign bounds can have a finite
radius even when their full width overflows. Point midpoints preserve the smallest subnormal
when both positions are equal to it.

**Several scalar kernels change results near floating-point boundaries.** `Real::ldexp`
rounds once into the subnormal range. `Real::ln_exp_m1` preserves the small result near
`ln(2)`, and the `f64` exponential integral `Real::exp_int_ei` preserves finite results past
the overflow point of plain exponentiation. For example, `exp_int_ei(710.0)` is about
`3.1509156882062014e305`.

**`euclid_utils::assert_exact` now compares bits.** Opposite signs of zero compare unequal.
NaNs with identical bits compare equal. The endpoints returned by `sweep` also preserve
their original bits, including the sign of zero.

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
