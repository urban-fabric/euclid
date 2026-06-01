# euclid

`euclid` is a small Rust crate for domain-neutral quantity newtypes used across
Urban Fabric code. It keeps unit conversions and wire/storage representations in
one place without depending on application crates.

## Public Modules

- `length`, `time`, `speed`, and `angle` define floating-point physical units
  such as `LengthMeters`, `TimeSeconds`, `SpeedKph`, and `AngleRadians`.
- `factor` and `percentage` define dimensionless scalar quantities.
- `bytes` and `count` define integer newtypes for byte sizes, byte ranges,
  generic counts, and thread counts.
- `timestamp` defines Unix epoch timestamp wrappers.
- `weights` defines route-cost values with explicit infinity sentinels:
  `TimeWeight` stores centiseconds and `DistanceWeight` stores decimeters.
- `storage` defines raw persisted unit wrappers, such as `TimeCentiseconds` and
  `DistanceDecimeters`, without route-cost semantics.
- `algebra` provides dimensional arithmetic between units.

## Boundary

This crate is intentionally domain-agnostic. It should contain unit wrappers,
lossy quantization rules, serialization contracts, and basic dimensional
arithmetic. It should not contain application workflow helpers, graph-specific
state machines, or project-specific control-flow types.

`weights` and `storage` intentionally use some of the same raw units. Use
`weights` when a value participates in route-cost ordering or may use the
`u32::MAX` infinity sentinel. Use `storage` when the type represents a raw
centisecond or decimeter field in persisted data.

Quantity payloads are private. Construct values with `new`, `from_*`, or
conversion methods, and read values with named accessors such as `as_meters`,
`as_seconds`, `as_kph`, `as_centiseconds`, or `as_u64`.

## Validation

Run the normal Rust checks before publishing changes:

```sh
cargo fmt --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```
