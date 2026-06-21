# euclid

Domain-neutral numeric foundations for the urban-fabric math stack.

euclid defines the lazy numeric contracts used by scalar, interpolation, array,
random, and unit-aware math code. A formula written against these traits can evaluate on
`f64`/`f32` or be represented by graph/tensor values. euclid keeps the contract
surface small and mathematical: it describes the operations a formula may use,
not storage layout, runtime state, or concrete array implementations.

The crate has no external dependencies. It ships trait contracts, the `f64` and
`f32` reference scalar implementations, integer/natural/bool scalar carriers,
and lazy physical-unit wrappers.

```toml
[dependencies]
euclid = "0.1"
```

## Design Rules

- Lazy values are `Clone`, not `Copy`. A value may be a scalar, expression, or
  graph handle.
- Constants are lifted with methods such as `LazyReal::scalar`; they are not
  exposed as associated `const`s.
- Comparisons produce masks. Use `LazyCompare::select` or the branchless
  `min`/`max`/`clamp` selectors instead of relying on host `bool`.
- Traits expose mathematical results. For example, `LazyMatrix` has
  `solve_spd`, `logdet_spd`, and `mahalanobis_squared`, but no `inverse`.
- Operations are grouped by the structure they require. Code should ask for the
  smallest trait bound that contains the operations it uses.

## Example

```rust
use euclid::LazyReal;

/// Standard-normal log-density, generic over the lazy real surface.
fn standard_normal_logpdf<T: LazyReal>(x: T) -> T {
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
| [`random`](src/random.rs) | random-source contracts over euclid scalar and array types |
| [`units`](src/units/mod.rs) | lazy physical quantities over any `LazyReal` carrier |

## Numeric Traits

The numeric surface is graded by algebraic structure.

- `LazySemiring`: `Clone`, `+`, `*`, `zero`, and `one`.
- `LazyNatural`: natural-number operations on top of `LazySemiring`, including
  `div_rem`, conversion to integer/real carriers, and count combinatorics.
- `LazyRing`: `LazySemiring` plus subtraction and negation.
- `LazyInteger`: signed integer operations on top of `LazyRing`, including
  Euclidean `div_rem` and conversion to real carriers.
- `Arithmetic`: `LazyRing` plus division.
- `LazyReal`: real analysis over `Arithmetic`: lifted constants,
  transcendentals, square/powers, rounding/sign operations, comparison-free
  selectors, masks, special functions, and stable log-domain helpers.
- `VectorSpace`: addition/subtraction plus scalar multiplication over a real
  field.
- `InnerProductSpace`: explicit inner products over a `VectorSpace`.
- `NormedVectorSpace`: explicit squared norm and norm operations over an
  `InnerProductSpace`.
- `NormedDivisionAlgebra`: the real/complex/quaternionic surface used by vector
  and matrix algebra: conjugation, magnitude, real part, and real-axis lifting.
- `LazyComplex` and `LazyQuaternion`: construction and component access for the
  complex and quaternionic refinements.
- `LazyBool` and `LazyCompare`: mask algebra, comparisons, and `select`.

`LazyReal` includes probability/statistics primitives such as `erf`, `erfc`,
`erf_inv`, `erfcx`, `ln_gamma`, `digamma`, `trigamma`, `gamma_lr`,
`gamma_ur`, `beta_reg`, `bessel_i0e`, `bessel_i1e`, `xlogy`, and `xlog1py`.
The `f64` implementations live in [`src/numeric/special`](src/numeric/special/mod.rs);
the `f32` implementations live in [`src/numeric/special/f32`](src/numeric/special/f32/mod.rs).
Those modules are crate-internal; the public API is the trait surface.

## Array Traits

The array surface is split by rank.

| rank | trait | requires | adds |
|------|-------|----------|------|
| 0 | `NormedDivisionAlgebra` | `Arithmetic` | conjugate, magnitude, real part |
| 1 | `LazyVector` | field-like scalar | constructors, vector-space ops, dot, norms, control flow |
| 2 | `LazyMatrix` | field-like scalar | constructors, matmul, SPD linear-algebra results, control flow |
| n | `LazyTensor` | `Clone` | shape, reshape, structural ops, reductions, contraction, control flow |

`LazyTensor` starts at `Clone` so structural tensor operations can work for
real, complex, integer, and boolean tensors. Arithmetic methods add the bounds
they need: `sum` and contraction require `LazyRing`, `prod` requires
`LazySemiring`, `min`/`max`/`mean` require `LazyReal`, and `any`/`all` require
`LazyBool`.

`LazyVector` and `LazyMatrix` use `NormedDivisionAlgebra` so real and complex
linear algebra share one contract. `dot` is Hermitian, `cholesky` is the
Hermitian positive-definite factorization, and real-valued results such as
`norm`, `logdet_spd`, and `mahalanobis_squared` return the scalar's real part.

## Interpolation

`Interpolate` defines scalar-parameterized interpolation with explicit `lerp`
and `midpoint` methods. euclid does not derive midpoint from `lerp`; each
implementation supplies the numerics it wants.

The module also provides `sample_at` for natural-number sample indices and
`sample` for endpoint-inclusive uniform samples.

## Random Traits

The random module defines stream-shaped and keyed random APIs:

- `RandomStream` and `TensorRandomStream`: mutable streams for uniform, normal,
  and gamma draws as scalars, vectors, matrices, and tensors.
- `KeyedRandomBackend`: stateless counter/key style draws, suitable for
  reproducible graph state.
- `SampleWith` and `RandomProcess`: sampling and stream-stateful processes.

The associated scalar, vector, matrix, and tensor types are euclid contracts.
The module does not prescribe a concrete PRNG, storage type, or tensor runtime.

## Units

`units` contains lazy physical quantities such as `LazyMeters`, `LazySeconds`,
`LazyKm`, `LazySpeedMs`, and `LazySquareMeters`. Each wrapper is generic over a
`LazyReal` carrier, so the same unit type can hold a host scalar, expression, or
graph scalar.

These are the lazy peers of the concrete `Copy`/`Pod`/serde unit newtypes in
`oresme`. euclid keeps the dependency-free lazy form here because it only
requires the `LazyReal` surface.

## Implemented Here

- `f64` and `f32` implement `Arithmetic`, `LazyReal`,
  `NormedDivisionAlgebra`, `VectorSpace`, `InnerProductSpace`,
  `NormedVectorSpace`, and `Interpolate`.
- `i32`, `i64`, and `i128` implement `LazyInteger`.
- `u32`, `u64`, and `u128` implement `LazyNatural`.
- `bool` implements `LazyBool`.
- Lazy physical quantities implement the unit traits over any `LazyReal`
  carrier.

euclid defines array and random contracts but does not include concrete array,
tensor, or random-stream implementations.

> **History.** Earlier versions of euclid also carried the abstract-algebra
> tower and concrete `Copy` number tier. Those live in `napier` now. Concrete
> `Copy`/`Pod`/serde unit types live in `oresme`. euclid keeps the lazy,
> carrier-generic contracts.

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
