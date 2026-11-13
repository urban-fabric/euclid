# napier

**The abstract-algebra tower, and the concrete `Copy` numeric tier where it
meets euclid's analysis surface.**

euclid's other crates are contracts that deliberately say nothing about what a
number *is*: `euclid-numeric`'s `Real` is implemented by `f64`, but equally by
an expression handle or a tensor element that is neither `Copy` nor one machine
word. napier is the tier for code that genuinely needs the real number line —
eager geometry, matrix and physics kernels — plus the algebraic structures those
kernels are written over.

```rust
use napier::Real;

// Written once against the concrete tier: constants and `Copy` are available,
// so this is an ordinary scalar formula rather than a staged expression.
fn circle_area<T: Real>(r: T) -> T {
    T::PI * r * r
}

let a = circle_area(2.0_f64);
let b = circle_area(2.0_f32); // same formula, different carrier
```

## What is in it

| Module | What it holds |
|---|---|
| `algebra` | The tower: `Field` → `OrderedField` / `AlgebraicallyClosedField`, the semiring/dioid stack, and the exact carriers `Rational`, `Zn`/`Zp`, `Integer`, `GaussianRational`. |
| `Real` | `OrderedField` + euclid's `Real` + `Copy`, plus the associated constants (`PI`, `TAU`, `EPSILON`, `INFINITY`, …) and the `bool`-returning predicates (`is_finite`, `is_nan`, …). Implemented by `f64` and `f32`. |
| `Complex` | `AlgebraicallyClosedField` + euclid's `Complex` + `Copy`, plus the complex transcendentals. |

euclid's analysis traits are imported here under the names `LazyReal` /
`LazyComplex`. The "lazy" names the *contract* tier, not deferred evaluation — a
staged or columnar backend implements the same contracts and stays entirely on
that tier, never depending on this crate.

## Why it lives in euclid

napier was its own repository until it was folded here. Its only dependency was
euclid, nothing released it independently, and it had gone stale: it still named
the single `euclid` facade crate that euclid 0.3 removed in favour of per-tier
crates, so it no longer resolved against euclid's own main branch.

It keeps its own `Proprietary` licence rather than inheriting the workspace's
Apache-2.0 — a move should not relicense code.
