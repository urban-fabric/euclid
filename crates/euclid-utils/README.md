# euclid-utils

Shared numerical helpers for tests: comparison, calculus, sweeps, and shape
assertions. Part of [euclid](../..).

Dev-only (`publish = false`), on the same footing as `euclid-stability`: take it
under `[dev-dependencies]`. These crates are not on crates.io — see
[the root README](../../README.md#using-it) for the git dependency to add.

```rust
use euclid_utils::{assert_close, derivative, integrate, log_sweep};

// A closed form checked against a numerical slope.
assert_close(derivative(f64::sin, 0.3), 0.3_f64.cos(), 1e-8, "d/dx sin");

// A density that must integrate to one.
assert_close(integrate(|x| 2.0 * x, 0.0, 1.0, 64), 1.0, 1e-12, "∫2x over [0,1]");

// A domain spanning orders of magnitude, hit at both bounds exactly.
for x in log_sweep(1e-6, 1e6, 25) { /* ... */ }
```

## Why it exists

The same handful of helpers kept being rewritten — `rel` had four copies and
`close`/`agrees` five each, with slightly different tolerance conventions.
**Divergent comparison helpers are worse than duplicated ones: a test can look
strict and be loose.** One definition means one convention.

## The comparison convention

Everything here compares with a **mixed absolute/relative** criterion:

```text
|got − want| ≤ tol · max(1, |want|)
```

A purely relative test is wrong wherever the quantity passes through zero — `ln Γ`
at 1 and 2, `digamma` near 1.46, `erf` at 0 — because it compares two different
roundings of zero and reports a meaningless ratio. A purely absolute test is wrong
wherever the quantity is large. Scaling by `max(1, |want|)` degrades gracefully to
absolute near zero and to relative away from it.

Where a result should be *as accurate as the format allows*, prefer
`euclid_stability::tolerance::within_ulps` instead: ULP is scale-free and is the
unit the hardware and the literature both use.

## What is exact, and what is not

| helper | exactness |
|---|---|
| `derivative` | five-point stencil, `O(h⁴)`; **exact on polynomials up to degree 4** |
| `integrate` | composite Simpson, `O(h⁴)`; **exact on polynomials up to degree 3** |
| `sweep` | first and last points are **bit-equal** to the bounds |
| `log_sweep` | constant ratio between steps; bounds to round-off (`log10`/`powf` do not round-trip exactly) |

Those exactness classes are what make the quadrature testable without an oracle,
and `tests/numerics.rs` asserts them: a single mistyped weight breaks exactness
on a cubic while leaving `O(h⁴)` accuracy on `sin` looking roughly fine, so a
tolerance test against a named function would not notice.

`sweep`'s endpoint exactness is a correction rather than a given. It stepped as
`lo + step·i`, which rounds twice, and `sweep(8, -7.985505297780037, 12)` ended
two ulp short of its own bound. That matters more than two ulp usually would: a
suite sweeping *to* a boundary in order to check behaviour *at* that boundary
misses it entirely when the endpoint falls short — silently, because the
assertion still passes on the point just inside. It is now the convex form
`lo(1−t) + hi·t`, which is the endpoint law `euclid_numeric::Interpolate` states
for `lerp`.

## Why the name is broad

Deliberately broader than "test helpers". If a genuinely general-purpose utility
turns up later that does not belong in a contracts crate, this is where it goes —
at which point some consumer would move it out of `[dev-dependencies]`.

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
Copyright 2026 Raymond Mead.
