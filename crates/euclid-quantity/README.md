# euclid-quantity

`euclid-quantity` provides small, domain-neutral **physical-unit primitives**:
typed quantities, unit conversions, compact storage wrappers, and basic
dimensional arithmetic.

It is the **carrier** tier of euclid: the other euclid crates own the
carrier-generic numeric, tensor, and unit *contracts*, and euclid-quantity
supplies the concrete `f64` unit/storage types that implement them. It names the
two tiers it builds on directly — `euclid-units` for the dimension contracts and
the `Dimension`/`dim` vocabulary these units implement, and `euclid-scalar` for
the interval kernels `euclid_quantity::ops::Interval` delegates its arithmetic
to.

euclid-quantity is also the one euclid crate that is not dependency-free: it
carries `bytemuck` for the `Pod`/`Zeroable` storage wrappers and `serde`. That
cost is per-crate — depending on `euclid-units` alone still pulls nothing
third-party.

## Public API

This crate's own units come from its root; the dimension *contracts* they
implement come from `euclid-units`, which defines them. Nothing is re-exported
across the crate boundary, so each name is imported from the crate that owns it:

```rust
use euclid_quantity::{LengthMeters, SpeedMs, TimeSeconds};
use euclid_units::Length;

let distance = LengthMeters::new(100.0);
let elapsed = TimeSeconds::new(10.0);
let speed: SpeedMs = distance.per_time(elapsed);

assert_eq!(speed.as_ms(), 10.0);
```

Use structured module paths when you want the implementation layout:

```rust
use euclid_quantity::conversion::UnitOfMeasure;
use euclid_quantity::ops::Interval;
use euclid_quantity::units::geometry::LengthMeters;
use euclid_units::dim;
```

The crate is organized by responsibility:

- `euclid_quantity::quantity` adds the two things euclid's quantity contracts
  cannot express — `HostQuantity` and `QuantityError`. It defines no contracts of
  its own: `Dimension`, `Measure`, `Quantity`, `ScalarQuantity`,
  `LinearQuantity` and `DiscreteQuantity` all live in `euclid_units` and are
  imported from there, not through this crate.
- `euclid_quantity::conversion` defines `UnitOfMeasure`. The type-level `dim`
  markers are euclid's, so they come from `euclid_units::dim` — a dimension has
  one definition and one path to it.
- `euclid_quantity::units` holds the physical units, grouped by euclid dimension
  family — `geometry` (length, area, volume, angle), `kinematics` (time, speed,
  acceleration, frequency, …), `mechanics` (mass, force, energy, power, …),
  `thermal`, `electromagnetism`, `chemistry`, and `dimensionless` (factor,
  percentage). Each family co-locates its unit newtypes with their
  `UnitOfMeasure` conversions and their implementations of euclid's base and
  dimension-specific contracts.
- `euclid_quantity::ops` contains quantity operations: affine points and
  uncertainty intervals.
- `euclid_quantity::data` contains boundary/storage wrappers: bytes, counts,
  epoch timestamps, centiseconds, and decimeters.

euclid's dimension traits are **not** re-exported here. The algebra in the
[dimensional algebra](#dimensional-algebra) example needs `euclid_units::Length`
in scope, and so a direct dependency on `euclid-units`. That is deliberate: a
re-export would put a second, undocumented path on a trait euclid owns, and hide
which crate a caller actually depends on.

## Conversion

Every scalar unit declares its dimension and how to express itself in that
dimension's primary unit. `UnitOfMeasure::convert` works only between units with
the same type-level dimension.

```rust
use euclid_quantity::{LengthFeet, LengthKm, LengthMeters, UnitOfMeasure};

let distance = LengthKm::new(1.0);
let meters: LengthMeters = distance.convert();
let feet: LengthFeet = distance.convert();

assert_eq!(meters.as_meters(), 1000.0);
assert!((feet.as_feet() - 3280.839_895).abs() < 1e-6);
```

Temperature conversion is offset-aware, so absolute temperatures can convert
between Kelvin, Celsius, and Fahrenheit without becoming vector quantities.

## Arithmetic

`ScalarQuantity` provides shared helpers for ordering, rounding, finiteness
checks, and typed approximate equality. These are euclid's traits, so the
predicates are typed to return a carrier mask — but `<f64 as Compare>::Mask` is
`bool`, so for every unit here they return a plain `bool`.

`LinearQuantity` marks quantities that behave like displacements. Linear
quantities can be added, subtracted, negated, scaled, interpolated, summed, and
used in `Interval<Q>` and `Point<Q>`.

Absolute temperatures intentionally do not implement `LinearQuantity`, which is
what keeps them out of `Interval` and `Point`.

`HostQuantity` carries the two conveniences euclid's carrier-generic contracts
structurally cannot: `from_raw_finite`, which returns a `Result` and so must
branch on the value, and `map`, which applies a host `Fn(f64) -> f64`. A carrier
that might be an expression handle can do neither. Every unit here is a concrete
`f64`, so both are available by blanket impl.

## Dimensional algebra

Cross-dimensional products and quotients come from euclid's dimension contracts,
which euclid-quantity's primary units implement: a length over a time is a
speed, a mass times an acceleration is a force, and so on across the SI web. The
algebra is exposed as methods on euclid's traits (`Length::per_time`,
`Mass::times_acceleration`, ...) rather than as `/` and `*` operators:

```rust
use euclid_quantity::{LengthKm, LengthMeters, SpeedMs, TimeSeconds, UnitOfMeasure};
use euclid_units::Length;

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

`euclid_quantity::data` contains compact, integer-backed wrappers for values
that cross storage, wire, or system boundaries:

- `Bytes` and `ByteRange` (information)
- `Count` and `ThreadCount` (dimensionless counts)
- `UnixSeconds` and `UnixMillis` (epoch instants)
- `TimeCentiseconds` and `DistanceDecimeters` (compact time/length storage)

Because their carriers are integers (`u64`/`u32`/`i64`) rather than `f64`, they
cannot be a euclid `Quantity` (which requires a `Real` carrier), but they still
join the same contract system through euclid's carrier-generic base trait:

- every one implements `euclid_units::Measure` — a runtime `Dimension` tag, unit
  symbol, and `raw`/`from_raw`, over any numeric carrier;
- the counts and sizes (`Count`, `ThreadCount`, `Bytes`) implement
  `euclid_units::DiscreteQuantity` for semiring `add` / saturating subtraction;
- the epoch timestamps (`UnixSeconds`, `UnixMillis`) implement
  `euclid_units::Instant` — the affine `since` / `advance` of an absolute
  position. `Bytes` carries the dedicated `Dimension::Information`.

These types name raw values and serialization contracts. Route-cost sentinels,
workflow state, graph-specific ordering, and application control-flow types
belong in the crates that use `euclid-quantity`.

## Adding Units

Each unit lives in a file under `src/units/<family>/`, such as
`units/mechanics/force.rs`. Units for the same quantity share a subdirectory
and conversion constants, as in `units/geometry/length/`. The workspace uses
`<module>.rs` beside the corresponding directory. Copy the nearest unit's
file to follow its implementation pattern.

1. Create the unit file with the transparent `f64` newtype
   (`pub(crate)` field, standard derives), a `new` constructor, a raw accessor,
   and any helpers.
2. Implement `euclid_units::Measure` — the carrier type, runtime `Dimension`,
   unit symbol, and `raw`/`from_raw`. This is the only one with a body to write.
3. Opt into the contracts the unit actually satisfies with **empty impl
   blocks**: `euclid_units::ScalarQuantity` always, and
   `euclid_units::LinearQuantity` only for displacement-like quantities, never
   an absolute scalar like temperature. Both are defaulted over the carrier's
   `Real` surface, so there is no body — the empty block is just the claim.
   `Quantity` needs no impl at all; it is blanket-implemented for every
   `Real`-backed `Measure`.
4. Implement `conversion::UnitOfMeasure` (scale/offset to the dimension's
   primary unit) and `core::fmt::Display` (delegating to
   `conversion::fmt_quantity`).
5. If it is a new dimension's primary unit, implement euclid's dimension
   contract for it and wire its associated types to the other primaries.
6. Implement the `+`/`-`/scaling/`Sum` operators (delegating to the
   `crate::quantity::scalar_*` helpers) if it is linear.
7. Declare the module and re-export the new type in the parent module. The
   family re-exports grouped units, and `units` re-exports each family at the
   crate root. Use the test index in `src/units/tests.rs` to find the relevant
   conformance, dimensional arithmetic, inherent method, and conversion tests.

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
