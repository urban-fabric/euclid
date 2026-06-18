# euclid

`euclid` provides small, domain-neutral primitives for algebra, scalar
computation, and typed quantities. It is the place for units, conversions,
compact storage wrappers, and basic dimensional arithmetic; it deliberately
does not depend on application crates.

## Public API

The public API has three top-level modules:

- `euclid::algebra` for semirings, dioids, rings, fields, and related algebraic
  traits.
- `euclid::numeric` for generic scalar traits such as `Scalar`, `LazyReal`,
  `LazyComplex`, `Real`, and `Complex`.
- `euclid::units` for quantities, unit conversion, data wrappers, and typed
  quantity operations.

`LazyReal` is the formula substrate for scalar-valued lazy backends — the whole
*elementwise analysis surface*, designed so a formula written once runs on `f64`
today and lowers to a tensor backend (eventually XLA) tomorrow. It carries:

- the **transcendentals** — `exp`/`ln`/`sqrt`, the trig family (`sin`/`cos`/`tan`,
  `atan2`/`hypot`, the hyperbolics), `powf`/`powi`/`cbrt`, rounding and sign;
- the comparison-free **selectors** `min`/`max`/`clamp` (a tensor lowers these to
  `Max`/`Min`/`Clamp`, so piecewise formulas stay portable without a `bool`);
- the **special functions** `erf`/`erfc`/`erf_inv`/`erfcx`, `ln_gamma`/`digamma`/
  `trigamma`, the regularized incomplete gamma `gamma_lr`/`gamma_ur`, the
  regularized incomplete beta `beta_reg`, the scaled modified Bessel
  `bessel_i0e`/`bessel_i1e`, and the stable `ln_1m_exp` — the CDF / entropy /
  divergence kernels of probability theory, each with a direct tensor-op analogue.
  Their concrete `f64` reference lives in `euclid::special`;
- a library of lifted **constants** — `half()`, `pi()`, `half_ln_tau()`, Euler's
  number `e()`, the log-base conversions (`log2_e()`, `log10_e()`, `ln_10()`,
  `log2_10()`, `log10_2()`), and the Euler–Mascheroni constant `euler_gamma()`.

The concrete `Real` tier keeps only what a tensor *cannot* produce — the associated
constants (`PI`/`TAU`/`SQRT_2`/`E`/…) and the `bool`-returning predicates
(`is_finite`/`is_nan`/`is_sign_negative`). Vector, tensor, and shaped construction
stays in the backend that owns those containers.

Common quantity types and traits are also re-exported from the crate root:

```rust
use euclid::{LengthMeters, SpeedMs, TimeSeconds, UnitOfMeasure};

let distance = LengthMeters::new(100.0);
let elapsed = TimeSeconds::new(10.0);
let speed: SpeedMs = distance / elapsed;

assert_eq!(speed.as_ms(), 10.0);
```

Use structured module paths when you want the implementation layout:

```rust
use euclid::units::conversion::{UnitOfMeasure, dim};
use euclid::units::measure::length::LengthMeters;
use euclid::units::ops::Interval;
```

## Units

The `units` module is organized by responsibility:

- `units::quantity` defines `Dimension`, `Quantity`, `ScalarQuantity`,
  `LinearQuantity`, and `QuantityError`.
- `units::conversion` defines `UnitOfMeasure` and the type-level `dim` markers.
- `units::measure` contains base measurements: length, time, angle, mass,
  temperature, factor, and percentage.
- `units::derived` contains derived measurements: speed, area, and acceleration.
- `units::ops` contains quantity operations: dimensional arithmetic, affine
  points, and uncertainty intervals.
- `units::data` contains boundary/storage wrappers: bytes, counts, epoch
  timestamps, centiseconds, and decimeters.

## Conversion

Every scalar unit declares its dimension and how to express itself in that
dimension's primary unit. `UnitOfMeasure::convert` works only between units with
the same type-level dimension.

```rust
use euclid::{LengthFeet, LengthKm, LengthMeters, UnitOfMeasure};

let distance = LengthKm::new(1.0);
let meters: LengthMeters = distance.convert();
let feet: LengthFeet = distance.convert();

assert_eq!(meters.as_meters(), 1000.0);
assert!((feet.as_feet() - 3280.839_895).abs() < 1e-6);
```

Temperature conversion is offset-aware, so absolute temperatures can convert
between Kelvin, Celsius, and Fahrenheit without becoming vector quantities.

## Arithmetic

`ScalarQuantity` provides shared helpers for finite construction, ordering,
rounding, finiteness checks, and typed approximate equality.

`LinearQuantity` marks quantities that behave like displacements. Linear
quantities can be added, subtracted, negated, scaled, interpolated, summed, and
used in `Interval<Q>` and `Point<Q>`.

Absolute temperatures intentionally do not implement `LinearQuantity`.

Dimensional operators are intentionally defined for primary units only:

```rust
use euclid::{LengthKm, LengthMeters, SpeedMs, TimeSeconds, UnitOfMeasure};

let distance: LengthMeters = LengthKm::new(5.0).convert();
let elapsed = TimeSeconds::new(1800.0);
let speed: SpeedMs = distance / elapsed;

assert!((speed.as_ms() - 2.777_777_777_777_777_7).abs() < 1e-12);
```

Convert non-primary units before dimensional arithmetic. This keeps the operator
surface small and avoids a combinatorial set of unit-pair implementations.

## Boundary Types

`units::data` contains compact wrappers for values that cross storage, wire, or
system boundaries:

- `Bytes` and `ByteRange`
- `Count` and `ThreadCount`
- `UnixSeconds` and `UnixMillis`
- `TimeCentiseconds` and `DistanceDecimeters`

These types name raw values and serialization contracts. Route-cost sentinels,
workflow state, graph-specific ordering, and application control-flow types
belong in the crates that use `euclid`.

## Adding Units

When adding a scalar unit:

1. Add a transparent newtype under `units::measure` or `units::derived`.
2. Implement `Quantity` with a private payload, named constructor, accessor,
   runtime dimension, and unit symbol.
3. Implement `LinearQuantity` only for displacement-like quantities.
4. Add the `UnitOfMeasure` implementation in `units::conversion`.
5. Add same-unit operator support if the type is linear.
6. Re-export the type from its local module, `units`, and the crate root if it
   should be part of the ergonomic API.
7. Add conversion, arithmetic, serde, and boundary tests matching nearby
   modules.

## Validation

Run the standard checks before publishing changes:

```sh
cargo fmt --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --doc
```
