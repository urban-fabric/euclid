# euclid

**Domain-neutral numeric foundations — the backend-neutral contract layer for the
urban-fabric math stack.**

euclid is a set of trait *contracts*, and almost nothing more. Its job is to let a
numeric formula be written **once** and run on **many backends**: the same Gaussian
log-density, Cholesky solve, or Mahalanobis distance you write against these traits
evaluates eagerly on `f64`/`f32` today and lowers to a tensor compiler (XLA, via
[`maxwell`](#ecosystem)) tomorrow — with no change to the formula.

The crate is **pure trait contracts plus the `f64`/`f32` reference scalars**, and has
**no external dependencies**.

```toml
[dependencies]
euclid = "0.1"
```

## The idea: lazy, write-once numerics

Every contract in euclid is *lazy*: it is `Clone` (never `Copy`) and carries only
operations a tensor backend can actually produce. That single constraint is what makes
a formula portable:

- **No scalar `bool`.** A tensor comparison yields a *mask*, not a boolean, so the
  comparison/`select` surface (`LazyBool` / `LazyCompare`) is its own tier and piecewise
  logic is expressed with the comparison-free `min` / `max` / `clamp` selectors.
- **Constants are lifted, not read.** `LazyReal::scalar(x)` lifts a host `f64` into the
  backend (an eager value, or a constant graph node) instead of reading an associated
  `const` a tensor type cannot expose.
- **Operations map op-for-op.** A formula that stays inside this surface lowers directly:
  `exp → Exp`, `erf → Erf`, `dot → DotGeneral`, `cholesky → Cholesky`, and so on.

Write the kernel against the trait; pick the carrier later.

```rust
use euclid::LazyReal;

/// Standard-normal log-density `−½x² − ½·ln(2π)`, written once over the lazy real
/// surface. Generic over `T: LazyReal`, so the *same* code runs on a concrete `f64`
/// today and on a tensor scalar tomorrow.
fn standard_normal_logpdf<T: LazyReal>(x: T) -> T {
    let x2 = x.clone() * x;
    -(T::half() * x2) - T::half_ln_tau()
}

// Evaluated eagerly here, because `f64: LazyReal`.
let lp = standard_normal_logpdf(0.0_f64);
assert!((lp - (-0.5 * (2.0 * std::f64::consts::PI).ln())).abs() < 1e-12);
```

## Public API

Three top-level modules:

| module | what it contracts |
|--------|-------------------|
| [`numeric`](src/numeric/mod.rs) | the scalar surface — the analysis and algebra of a single number |
| [`array`](src/array.rs) | the rank-graded array contracts — vector, matrix, tensor |
| [`random`](src/random.rs) | backend-neutral random-source contracts over those types |

### `numeric` — the scalar surface

The number traits, graded by structure so a formula demands exactly the operators it
uses, no more:

- **`LazySemiring`** — `Clone` + `+ *` (with identities `0`/`1`), subtraction and
  negation *withheld*. Blanket-implemented for every carrier with `Add` and `Mul`.
  - **`LazyNatural`** refines it with the natural surface (the machine-natural lift,
    floor `div_rem`, `to_real`, and `to_integer` — the widen into the signed ring where
    subtraction lives), plus count combinatorics (`factorial`, `ln_factorial`,
    `binomial`, `ln_binomial`) — the `u64`-style non-negative count/index/shape dtype.
    Implemented for `u32`/`u64`/`u128`.
  - **`LazyRing`** refines it with `-` and negation, with division *deliberately
    withheld*. Blanket-implemented for every carrier with the four ring operators.
    - **`Arithmetic`** refines it with `/` — the field-operator bound (`f64`/`f32`,
      complex, tensors).
    - **`LazyInteger`** refines it with the integer surface (the machine-integer lift,
      Euclidean `div_rem`, and `to_real`) — the `s64`-style count/index/shape dtype.
      Implemented for `i64`/`i128`.
- **`LazyReal`** — the real-number analysis surface: the transcendentals
  (`exp`/`ln`/`sqrt`, trig, `atan2`/`hypot`, hyperbolics), rounding/sign, the
  comparison-free `min`/`max`/`clamp` selectors, a library of lifted constants
  (`pi`, `e`, `half_ln_tau`, `euler_gamma`, the log-base factors, …), and the
  **special functions** that probability and statistics reach for —
  `erf`/`erfc`/`erf_inv`/`erfcx`, `ln_gamma`/`digamma`/`trigamma`, the regularized
  incomplete gamma `gamma_lr`/`gamma_ur`, the regularized incomplete beta `beta_reg`,
  the scaled modified Bessel `bessel_i0e`/`bessel_i1e`, `xlogy`, and scalar stable
  log-domain forms such as `ln_1p_exp`, `ln_1m_exp`, `ln_add_exp`, and `ln_sub_exp`.
  Each has a direct tensor-op analogue or lowers through those scalar primitives.
- **`NormedDivisionAlgebra`** — a normed division algebra over ℝ. By
  Frobenius/Hurwitz the associative ones are exactly **ℝ (1D), ℂ (2D), ℍ (4D)**, so one
  matrix/vector kernel runs over all three. It names the real field that magnitudes and
  norms live in, and the `conjugate` every Hermitian formula needs (the identity on a
  real number, so real kernels specialize for free).
  - **`LazyComplex`** adds the imaginary part (`re`/`im`/`conj`).
  - **`LazyQuaternion`** adds the `i`/`j`/`k` components (ℍ is non-commutative — a
    division ring, not a field).
- **`LazyBool` + `LazyCompare`** — the **mask** tier. Booleans are a Boolean algebra,
  not a ring, so they sit *beside* the tower: a comparison returns the abstract
  `LazyCompare::Mask` (`bool` eagerly, a `pred` tensor on a backend) and `select`
  consumes it.

The concrete `f64` implementations of the special functions live in the crate-internal
[`src/numeric/special`](src/numeric/special/mod.rs) tree (Lanczos gamma,
Numerical-Recipes continued fractions for the incomplete gamma/beta, Giles's inverse-erf
rational). A parallel internal [`f32`](src/numeric/special/f32/mod.rs) tree mirrors them
natively in single precision (same algorithms, `f32` throughout, convergence tolerances at
the `f32` noise floor), so the `f32` `LazyReal` impl evaluates each one without a `f64`
round-trip. They are not re-exported; the public API is the numeric trait surface.

### `array` — the rank-graded contracts

The array-level peers of the scalar surface. These are **abstract mathematical
contracts** — the operations a kernel is written in, not any backend's parameterization
of them (axis-naming, `uplo`/`trans` flags, memory layout, and precision policy stay
with the implementor).

| rank | trait | bound | adds |
|------|-------|-------|------|
| 0 | `NormedDivisionAlgebra` | — | conjugate + magnitude + the real field |
| 1 | `LazyVector` | field | inner product, norms, vector-space ops |
| 2 | `LazyMatrix` | field | matmul + the SPD linear-algebra quantities |
| n | `LazyTensor` | `LazyRing` | shape + reductions + contraction (real/complex/integer) |

Two bands, mirroring XLA/JAX's `lax`:

- **Dtype-agnostic structure.** `LazyTensor: LazyRing`, so shape/reshape/reductions/
  contraction run over real, complex, **or** integer elements alike (an `s64`
  index/count array is a first-class `LazyTensor`). Element kind is a *refinement*: a
  real tensor also implements `LazyReal`, a complex one `LazyComplex`, an integer one
  `LazyInteger`.
- **Field-only linear algebra.** `LazyVector`/`LazyMatrix` stay generic over a
  `NormedDivisionAlgebra` because inner products, norms, and SPD quantities genuinely
  need a field. `dot` is the **Hermitian** inner product `Σ conj(aᵢ)·bᵢ`; `cholesky`
  is the Hermitian factorization `Σ = L·Lᴴ` — both collapsing to the ordinary real case
  since `conjugate` is the identity there. The intrinsically-real outputs
  (`norm`, `logdet_spd`, `mahalanobis_squared`) return the scalar's real part even over ℂ.

`LazyMatrix` exposes only stated mathematical *results* — `solve_spd`, `solve_spd_vec`,
`logdet_spd`, `mahalanobis_squared` — and deliberately **no `inverse`**; the backend
owns the stable factorization. Richer decompositions and prepared-factor seams
(`SpdFactor`, `PreparedSpd`, …) live in [`frobenius`](#ecosystem).

### `random` — backend-neutral random sources

Contracts for drawing `Uniform(0,1)`, standard-normal, and gamma variates as lazy
scalars / vectors / matrices / tensors, independent of host `rand`:

- **`RandomStream`** / **`TensorRandomStream`** — a mutable stream (an eager adapter
  wraps a host `rand::Rng`; an XLA adapter advances an owned key).
- **`KeyedRandomBackend`** — the stateless, counter-based sibling (ThreeFry / Philox)
  where each call takes and returns an explicit key, for reproducible graph state.
- **`SampleWith`** / **`RandomProcess`** — a distribution sampled from a stream, and a
  stream-stateful stochastic process.

The associated scalar/vector/matrix/tensor types are euclid's own lazy contracts, so
nothing here names dense storage or a concrete backend.

## What's implemented where

euclid ships the **contracts** plus the **reference scalar carriers**:

- `f64` / `f32` implement `Arithmetic`, `LazyReal`, and `NormedDivisionAlgebra`.
- `i64` / `i128` implement `LazyInteger`.
- `u32` / `u64` / `u128` implement `LazyNatural`.
- `bool` implements `LazyBool`.

The array and random contracts are implemented by the backends downstream.

## Ecosystem

euclid owns the contracts; the implementations live in sibling crates:

- **`napier`** — the concrete `Copy` number tier (`Real` / `Complex`) and the
  abstract-algebra tower (semiring / dioid / ring / field + the exact carriers
  `Rational` / `Zn` / `GaussianRational`), formed by fusing this lazy surface with the
  algebraic field.
- **`frobenius`** — backend-generic linear-algebra algorithms and the richer matrix
  extension traits over `LazyMatrix`, for both eager and XLA backends.
- **`maxwell`** — the XLA / StableHLO compute backend: a euclid-contract implementation
  over compiled HLO, so write-once kernels run on accelerators.
- **`oresme`** — physical units, quantities, and conversions.
- **`bayes`** — closed-form probability and statistics (distributions, special
  functions, sample statistics, stochastic processes, conjugate Bayesian updates)
  written against this surface.

> **History.** Earlier versions of euclid also carried the abstract-algebra tower, the
> concrete `Copy` number tier, and a `units` module. Those have moved to `napier` and
> `oresme`; euclid is now strictly the contract layer.

## Validation

Run the standard checks before publishing changes:

```sh
cargo fmt --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --doc
```

## License

Proprietary. © Raymond Mead.
