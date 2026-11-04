# euclid-numeric

Scalar numeric contracts — the foundation of [euclid](../..) and of the Urban
Fabric numeric vocabulary. No dependencies.

```toml
[dependencies]
euclid-numeric = "0.3"
```

```rust
use euclid_numeric::Real;

/// Standard-normal log-density over any real carrier.
fn standard_normal_logpdf<T: Real>(x: T) -> T {
    let x2 = x.square();
    -(T::half() * x2) - T::half_ln_tau()
}
```

## Trait Tiers

Layered by algebraic structure. Pick the first tier that has the operation you
need.

| trait | adds |
|-------|------|
| `Element` | carrier-defined `zero`/`one`; shared by real, integer, natural, and boolean carriers |
| `Semiring` | `Element` plus `+` and `*` |
| `Natural` | unsigned count/index ops on `Semiring`: `div_rem`, conversions, count combinatorics |
| `Ring` | `Semiring` plus subtraction and negation |
| `Integer` | signed count/index ops on `Ring`: Euclidean `div_rem`, conversion to real carriers |
| `Arithmetic` | `Ring` plus division |
| `Real` | real analysis over `Arithmetic`: lifted constants, powers, transcendentals, rounding/sign, masks, selectors, special functions, stable log-domain helpers |
| `NormedDivisionAlgebra` | the ℝ/ℂ/ℍ surface: conjugation, magnitude, real part, real-axis lifting |
| `Complex`, `Quaternion` | construction and component access for those refinements |
| `Boolean`, `Compare` | mask algebra, comparisons, and `select` |

`Real` also carries probability/statistics primitives: `erf`, `erfc`, `erf_inv`,
`erfcx`, `ln_gamma`, `digamma`, `trigamma`, `gamma_lr`, `gamma_ur`, `beta_reg`,
`bessel_i0e`, `bessel_i1e`, `xlogy`, `xlog1py`.

Nothing here returns a scalar `bool` from a generic comparison; comparisons
return masks, consumed by `Compare::select` or `Compare::if_then_else`.

## Also Here

- **`interpolate`** — `Interpolate` defines scalar-parameterized interpolation
  with explicit `lerp` and `midpoint`. `midpoint` is required separately because
  some carriers need a more stable formula than `lerp(a, b, 0.5)`. It is
  deliberately opt-in rather than blanket-implemented over the algebra tiers: the
  affine `lerp` is the wrong interpolation for some division algebras (a unit
  quaternion wants `slerp`).
- **`control_flow`** — `ControlFlow: Compare` is the explicit graph
  conditional/loop contract, for carriers that can represent `cond` and
  `while_loop` as graph nodes.

Both live here rather than in crates of their own: `Interpolate`'s `f64`/`f32`
impls are part of this crate's reference implementations, and `ControlFlow`
refines `Compare`. Both are re-exported at the crate root.

## Implemented Here

- `f64`, `f32` — `Element`, `Arithmetic`, `Real`, `NormedDivisionAlgebra`,
  `Interpolate`
- `i32`, `i64`, `i128` — `Element`, `Integer`
- `u32`, `u64`, `u128` — `Element`, `Natural`
- `bool` — `Element`, `Boolean`

Concrete complex and quaternion carriers are not shipped; those traits are
contracts for consumer-provided types.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2026 Raymond Mead.
