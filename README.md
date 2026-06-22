# euclid

euclid is a dependency-free set of numeric contracts for writing carrier-generic
math in Rust.

Use it when a formula should work with ordinary `f64`/`f32` values and also with
your own expression, graph, tensor, or unit-aware carriers. euclid defines the
operations a formula may use; it does not define storage layout, execution
runtime, array allocation, or a random-number engine.

The crate includes:

- scalar traits for algebra, real analysis, masks, integer/natural counts, and
  complex/quaternion-style carriers;
- `f64`, `f32`, signed-integer, unsigned-integer, and `bool` implementations;
- rank-graded vector, matrix, and tensor traits;
- interpolation, random-source, and physical-unit traits.

```toml
[dependencies]
euclid = "0.2"
```

## How To Use It

- Choose the narrowest trait bound that contains the operations your formula
  needs. For example, use `Ring` for `+ - *`, `Real` for transcendental
  functions, and `Compare` only when you need masks or `select`.
- Treat generic values as `Clone`, not `Copy`. A carrier may be a host scalar,
  an expression handle, or a tensor value.
- Lift numeric literals through methods such as `Real::scalar`,
  `Real::half`, and `Real::scalar_like`; do not assume associated constants can
  represent every carrier.
- Comparisons return masks, not necessarily host `bool`. Use
  `Compare::select`, `Compare::if_then_else`, or the `Real::min`/`max`/`clamp`
  helpers instead of branching on a generic comparison.
- Matrix traits expose stable mathematical operations such as `solve_spd`,
  `logdet_spd`, and `mahalanobis_squared`; they intentionally do not expose an
  `inverse` method.

## Example

```rust
use euclid::Real;

/// Standard-normal log-density over any real carrier.
fn standard_normal_logpdf<T: Real>(x: T) -> T {
    let x2 = x.square();
    -(T::half() * x2) - T::half_ln_tau()
}

let lp = standard_normal_logpdf(0.0_f64);
assert!((lp - (-0.5 * (2.0 * std::f64::consts::PI).ln())).abs() < 1e-12);
```

## Public API

| module | purpose |
|--------|---------|
| [`numeric`](src/numeric/mod.rs) | scalar algebra, analysis, masks, and special functions |
| [`interpolate`](src/interpolate.rs) | scalar-parameterized interpolation and sampling helpers |
| [`array`](src/array/mod.rs) | rank-graded vector, matrix, and tensor contracts |
| [`random`](src/random.rs) | JAX-style keyed random-source contracts |
| [`units`](src/units/mod.rs) | physical quantities over any `Real` carrier |

## Numeric Traits

The numeric traits are layered by algebraic structure. Pick the first layer that
has the operation you need.

- `Element`: carrier-defined `zero` and `one`. This is shared by numeric,
  integer, natural, and boolean carriers.
- `Semiring`: `Element` plus `+` and `*`.
- `Natural`: unsigned count/index operations on top of `Semiring`, including
  `div_rem`, conversion to integer/real carriers, and count combinatorics.
- `Ring`: `Semiring` plus subtraction and negation.
- `Integer`: signed count/index operations on top of `Ring`, including
  Euclidean `div_rem` and conversion to real carriers.
- `Arithmetic`: `Ring` plus division.
- `Real`: real analysis over `Arithmetic`: lifted constants, powers,
  transcendentals, rounding/sign operations, masks, selectors, special
  functions, and stable log-domain helpers.
- `VectorSpace`: addition/subtraction plus scalar multiplication over a real
  field.
- `InnerProductSpace`: explicit inner products over a `VectorSpace`.
- `NormedVectorSpace`: explicit squared norm and norm operations over an
  `InnerProductSpace`.
- `NormedDivisionAlgebra`: the real/complex/quaternionic surface used by vector
  and matrix algebra: conjugation, magnitude, real part, and real-axis lifting.
- `Complex` and `Quaternion`: construction and component access for the
  complex and quaternionic refinements.
- `Boolean` and `Compare`: mask algebra, comparisons, and `select`.

`Real` also includes probability/statistics primitives such as `erf`, `erfc`,
`erf_inv`, `erfcx`, `ln_gamma`, `digamma`, `trigamma`, `gamma_lr`, `gamma_ur`,
`beta_reg`, `bessel_i0e`, `bessel_i1e`, `xlogy`, and `xlog1py`.

## Array Traits

The array surface is split by rank.

| rank | trait | requires | adds |
|------|-------|----------|------|
| 0 | `NormedDivisionAlgebra` | `Arithmetic` | conjugate, magnitude, real part |
| 1 | `Vector` | field-like scalar | constructors, vector-space ops, dot, norms, control flow |
| 2 | `Matrix` | field-like scalar | constructors, matmul, SPD linear-algebra results, control flow |
| n | `Tensor` | `Clone` | shape, reshape, structural ops, reductions, contraction, control flow |

`Tensor` starts at `Clone`, so structural operations can work for real,
complex, integer, natural, and boolean tensors. Arithmetic methods add the
bounds they need: `sum` and contraction require `Ring`, `prod` requires
`Semiring`, `min`/`max` require `Real`, `mean` requires
`NormedDivisionAlgebra`, and `any`/`all` require `Boolean`.

`Vector` and `Matrix` use `NormedDivisionAlgebra` so real and complex
linear algebra share one contract. `dot` is Hermitian, `cholesky` is the
Hermitian positive-definite factorization, and real-valued results such as
`norm`, `logdet_spd`, and `mahalanobis_squared` return the scalar's real part.

Beyond the core, opt-in **capability extension** traits grow the surface a
carrier can offer without widening the base contracts:

- `TensorReduce`: statistical and scan reductions — `var`, `std`,
  `logsumexp`, `softmax`, `log_softmax`, `cumsum`.
- `TensorSort`: order statistics — `sort`, `argsort`, `argmax`, `argmin`,
  `top_k` (index results use `Tensor::Index`).
- `TensorContract`: general contraction — `dot_general` and the `einsum1` /
  `einsum2` Einstein-summation forms.
- `TensorIndexing`: range and categorical encoding — `iota`, `one_hot`,
  `bincount`.
- `TensorTransform`: structural conveniences — `roll`, `tile`, `clip`.
- `MatrixDecompose`: dense factorizations and the general (non-SPD) solve —
  `qr`, `eigh`, `svd`, `lu`, `solve`, `solve_vec`.

euclid declares array traits but does not include a concrete vector, matrix, or
tensor type. Consumers provide implementations for the carrier types they want
to use.

## Interpolation

`Interpolate` defines scalar-parameterized interpolation with explicit `lerp`
and `midpoint` methods. `midpoint` is explicit because some carriers need a
different implementation than `lerp(a, b, 0.5)`.

The module also provides `sample_at` for natural-number sample indices and
`sample` for endpoint-inclusive uniform samples.

## Random Traits

The random module defines JAX-style keyed random APIs split by capability:

- `KeyedRandomBackend`: immutable key construction, serialization, splitting,
  and folding. It has no draw methods by itself.
- `KeyedRandomScalar`: scalar uniform, normal, gamma, raw bits, Bernoulli, and
  integer draws.
- `KeyedRandomMatVec`: rank-1/rank-2 real-array uniform, normal, and
  gamma draws.
- `KeyedRandomTensor`: arbitrary-rank real, boolean, signed-integer, and
  unsigned-integer tensor draws, including bits, Bernoulli, integer, and
  categorical sampling.
- `KeyedRandomScalarDistributions`, `KeyedRandomMatVecDistributions`, and
  `KeyedRandomTensorDistributions`: optional backend-supported samplers for
  distributions such as exponential, Gumbel, logistic, Cauchy, beta, Dirichlet,
  Poisson, permutation, and shuffle.
- `split` and `fold_in`: deterministic key derivation for independent draws,
  loop steps, chains, replicas, and batches.
- `key_data` and `wrap_key_data`: key serialization / interop hooks for typed
  and legacy key representations.
- `SampleWithKey` and `KeyedRandomProcess`: sampling and process-step contracts
  over explicit keys.

Implementors can support only the surfaces they actually have. Generic code can
require a distribution-extension trait when it wants the backend's native or
tuned sampler rather than composing a distribution from lower-level draws.
Batched keys should have map-like semantics: drawing from a batch of keys is
equivalent to drawing once per key and stacking the results. The module does not
prescribe a concrete PRNG, storage type, tensor runtime, or state representation.

## Units

`units` contains physical quantities such as `Meters`, `Seconds`, `Km`,
`SpeedMs`, and `SquareMeters`. Each wrapper is generic over a `Real` carrier,
so the same unit type can hold a host scalar, expression, graph scalar, or
tensor scalar.

## Implemented Here

- `f64` and `f32` implement `Element`, `Arithmetic`, `Real`,
  `NormedDivisionAlgebra`, and `Interpolate`.
- `i32`, `i64`, and `i128` implement `Element` and `Integer`.
- `u32`, `u64`, and `u128` implement `Element` and `Natural`.
- `bool` implements `Element` and `Boolean`.
- Physical quantities implement unit traits over any `Real` carrier.

euclid does not ship concrete vector, matrix, tensor, random backend, complex, or
quaternion implementations. Those traits are contracts for consumer-provided
carrier types.

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
