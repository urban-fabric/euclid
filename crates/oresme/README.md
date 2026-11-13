# oresme

`oresme` provides small, domain-neutral **physical-unit primitives**: typed
quantities, unit conversions, compact storage wrappers, and basic dimensional
arithmetic.

It is the **carrier** tier of euclid: the other euclid crates own the
carrier-generic numeric, tensor, and unit *contracts*, and oresme supplies the
concrete `f64` unit/storage types that implement them. It names the two tiers it
builds on directly — `euclid-units` for the dimension contracts and the
`Dimension`/`dim` vocabulary these units implement, and `euclid-numeric` for the
interval kernels `oresme::ops::Interval` delegates its arithmetic to.

oresme is also the one euclid crate that is not dependency-free: it carries
`bytemuck` for the `Pod`/`Zeroable` storage wrappers and `serde`. That cost is
per-crate — depending on `euclid-units` alone still pulls nothing third-party.

## Public API

Common quantity types and traits are re-exported from the crate root:

```rust
use oresme::{Length, LengthMeters, SpeedMs, TimeSeconds};

let distance = LengthMeters::new(100.0);
let elapsed = TimeSeconds::new(10.0);
let speed: SpeedMs = distance.per_time(elapsed);

assert_eq!(speed.as_ms(), 10.0);
```

Use structured module paths when you want the implementation layout:

```rust
use oresme::conversion::{UnitOfMeasure, dim};
use oresme::ops::Interval;
use oresme::units::geometry::LengthMeters;
```

The crate is organized by responsibility:

- `oresme::quantity` defines `Dimension`, `Quantity`, `ScalarQuantity`,
  `LinearQuantity`, and `QuantityError`.
- `oresme::conversion` defines `UnitOfMeasure` and re-exports euclid's type-level
  `dim` markers, so a dimension has one definition rather than two.
- `oresme::units` holds the physical units, grouped by euclid dimension family —
  `geometry` (length, area, volume, angle), `kinematics` (time, speed,
  acceleration, frequency, …), `mechanics` (mass, force, energy, power, …),
  `thermal`, `electromagnetism`, `chemistry`, and `dimensionless` (factor,
  percentage). Each family co-locates its unit newtypes with their
  `UnitOfMeasure` conversions and their implementations of euclid's base and
  dimension-specific contracts.
- `oresme::ops` contains quantity operations: affine points and uncertainty
  intervals.
- `oresme::data` contains boundary/storage wrappers: bytes, counts, epoch
  timestamps, centiseconds, and decimeters.

euclid's dimension traits are re-exported from the crate root, so the algebra in
the [dimensional algebra](#dimensional-algebra) example works with just
`use oresme::Length;` — no direct dependency on `euclid-units` needed.

## Conversion

Every scalar unit declares its dimension and how to express itself in that
dimension's primary unit. `UnitOfMeasure::convert` works only between units with
the same type-level dimension.

```rust
use oresme::{LengthFeet, LengthKm, LengthMeters, UnitOfMeasure};

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

## Dimensional algebra

Cross-dimensional products and quotients come from euclid's dimension contracts,
which oresme's primary units implement: a length over a time is a speed, a mass
times an acceleration is a force, and so on across the SI web. The algebra is
exposed as methods on euclid's traits (`Length::per_time`, `Mass::times_acceleration`,
...) rather than as `/` and `*` operators:

```rust
use oresme::{Length, LengthKm, LengthMeters, SpeedMs, TimeSeconds, UnitOfMeasure};

let distance: LengthMeters = LengthKm::new(5.0).convert();
let elapsed = TimeSeconds::new(1800.0);
let speed: SpeedMs = distance.per_time(elapsed);

assert!((speed.as_ms() - 2.777_777_777_777_777_7).abs() < 1e-12);
```

These contracts are implemented for each dimension's primary unit only
(`LengthMeters`, `TimeSeconds`, `SpeedMs`, ...), so convert non-primary units
first. That keeps the implementation linear in the number of dimensions instead
of combinatorial in the number of units.

## Boundary Types

`oresme::data` contains compact, integer-backed wrappers for values that cross
storage, wire, or system boundaries:

- `Bytes` and `ByteRange` (information)
- `Count` and `ThreadCount` (dimensionless counts)
- `UnixSeconds` and `UnixMillis` (epoch instants)
- `TimeCentiseconds` and `DistanceDecimeters` (compact time/length storage)

Because their carriers are integers (`u64`/`u32`/`i64`) rather than `f64`, they
cannot be a euclid `Quantity` (which requires a `Real` carrier), but they still
join the same contract system through euclid's carrier-generic base trait:

- every one implements `euclid_units::Measure` — a runtime `Dimension` tag,
  unit symbol, and `raw`/`from_raw`, over any numeric carrier;
- the counts and sizes (`Count`, `ThreadCount`, `Bytes`) implement
  `euclid_units::DiscreteQuantity` for semiring `add` / saturating subtraction;
- the epoch timestamps (`UnixSeconds`, `UnixMillis`) implement
  `euclid_units::Instant` — the affine `since` / `advance` of an absolute
  position. `Bytes` carries the dedicated `Dimension::Information`.

These types name raw values and serialization contracts. Route-cost sentinels,
workflow state, graph-specific ordering, and application control-flow types
belong in the crates that use `oresme`.

## Adding Units

Each unit lives in its own file under `src/units/<family>/` (one file per unit, e.g.
`units/mechanics/force.rs`); shared constants and the family's tests live in
`units/<family>/mod.rs`. The crate is deliberately macro-free, so each unit spells
out its impls explicitly; the quickest path is to copy the nearest existing unit's
file and adjust it:

1. Create `src/units/<family>/<unit>.rs` with the transparent `f64` newtype
   (`pub(crate)` field, standard derives), a `new` constructor, a raw accessor, and
   any helpers, then implement `crate::Quantity` (its runtime dimension and unit
   symbol) and `crate::LinearQuantity` — the latter only for displacement-like
   quantities, not absolute scalars like temperature.
2. Implement `conversion::UnitOfMeasure` (scale/offset to the dimension's primary
   unit) and `core::fmt::Display` (delegating to `conversion::fmt_quantity`).
3. Implement euclid's base contracts `euclid_units::{Quantity, ScalarQuantity}`
   (and `LinearQuantity` unless it is an absolute scalar), delegating to
   `<Self as crate::Quantity>::{raw, from_raw}`.
4. If it is a new dimension's primary unit, implement euclid's dimension contract
   for it and wire its associated types to the other primaries.
5. Implement the `+`/`-`/scaling/`Sum` operators (delegating to the
   `crate::quantity::scalar_*` helpers) if it is linear.
6. Declare the file in `units/<family>/mod.rs` (`mod <unit>; pub use <unit>::*;`);
   the family and `units::*` re-export it to the crate root automatically. Add
   tests in the family `mod.rs`'s `#[cfg(test)]` modules matching nearby ones.

## Validation

```sh
cargo fmt --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --doc
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE).

Apache-2.0 governs what a recipient may do with the code; it does not oblige
anyone to publish it. Keeping this repository private is a separate, purely
practical matter. Note the corollary: anyone who *does* receive it under
Apache-2.0 holds an irrevocable right to redistribute it, publicly included.
