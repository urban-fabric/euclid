# euclid-scalar

**The concrete `Copy` number line, where euclid's algebra tower and analysis
surface fuse.** Part of [euclid](../..).

These crates are not on crates.io. See [the root README](../../README.md#using-it)
for the git dependency to add.

euclid's contract crates deliberately say nothing about what a number *is*:
`euclid-numeric`'s `Real` is implemented by `f64`, but equally by an expression
handle or a tensor element that is neither `Copy` nor one machine word.
euclid-scalar is the tier for code that genuinely needs the real number line —
eager geometry, matrix and physics kernels.

```rust
use euclid_scalar::Real;

// Written once against the concrete tier: constants and `Copy` are available,
// so this is an ordinary scalar formula rather than a staged expression.
fn circle_area<T: Real>(r: T) -> T {
    T::PI * r * r
}

let a = circle_area(2.0_f64);
let b = circle_area(2.0_f32); // same formula, different carrier
```

## What is in it

| Trait | What it fuses |
|---|---|
| `Real` | `euclid_algebra::OrderedField` + `euclid_numeric::Real` + `Copy`, plus the associated constants (`PI`, `TAU`, `EPSILON`, `INFINITY`, …) and the `bool`-returning predicates (`is_finite`, `is_nan`, `is_sign_negative`). Implemented by `f64` and `f32`. |
| `Complex` | `euclid_algebra::AlgebraicallyClosedField` + `euclid_numeric::Complex` with a `Real` real part + `Copy`, plus the complex transcendentals. |

Two crates supply the halves: `euclid-algebra` says which algebraic tier a
carrier reaches, `euclid-numeric` says what operations a formula may use. This is
the only crate in the workspace that depends on both, which is precisely its job.

## Which bound to use

Prefer `euclid_numeric::Real`. It is the wider bound, and a formula written
against it runs on expression handles, graph nodes and tensor elements as well as
on `f64`.

| you need | bound |
|---|---|
| transcendentals, rounding, `min`/`max`/`clamp` — anything a tensor compiler can lower | `euclid_numeric::Real` |
| `Copy`, a host `bool` from a predicate, a single `PI` value | `euclid_scalar::Real` |

A `Tensor` or columnar `Expr` backend stays entirely on the contract tier and
never depends on this crate.

## Why it is a separate crate

Until 0.4 this tier lived inside `euclid-algebra`, which therefore owned two
goals at once: the abstract tower and the concrete number. The symptom was a name
collision — it imported euclid's `Real`/`Complex` as `LazyReal`/`LazyComplex` to
keep them apart from its own same-named traits, so the crate documented a
workaround instead of a boundary.

Splitting them removed the aliases. `euclid_numeric::Real` is the contract and
`euclid_scalar::Real` is the concrete number, so the distinction is carried by
the crate name rather than by an import rename. It also left `euclid-algebra`
with no dependencies at all.

## License

Apache-2.0, inherited from the workspace like every other member.

This crate was split out of `euclid-algebra`, which predates the fold into
euclid, and carried that crate's `Proprietary` licence until 0.4 — on the
reasoning that a split should not relicense any more than a move. That has been
overridden deliberately by the copyright holder: the fleet ships under one
licence. `cargo deny` is what surfaced it — `"Proprietary"` is not valid SPDX,
so the crate read as *unlicensed* rather than as restrictively licensed.
