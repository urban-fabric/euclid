# euclid

`euclid` is a small Rust crate for domain-neutral quantity newtypes used across
Urban Fabric code. It keeps unit conversions and wire/storage representations in
one place without depending on application crates.

## Public Layers

`euclid` is organized by responsibility, while the most common types are also
re-exported at the crate root for ergonomic imports.

- `core` defines the shared contracts: `Dimension`, `Quantity`,
  `ScalarQuantity`, and `UnitOfMeasure`. `core::units` provides the
  dimension-checked `convert` / `value_in` protocol, including offset-aware
  temperature conversion.
- `measure` defines base scalar measurements: `length`, `time`, `angle`,
  `mass`, `temperature`, plus dimensionless `factor` and `percentage`.
- `derived` defines measurements derived from base dimensions: `speed`, `area`,
  and `acceleration`.
- `math` defines operations over quantities: same-unit scalar operators,
  dimensional `algebra`, affine `Point<Q>`, and uncertainty `Interval<Q>`.
- `data` defines storage and boundary-oriented wrappers: `bytes`, `count`,
  `timestamp`, raw persisted `storage`, route-cost `weights`, and checked
  lossy `quantize` helpers.

Compatibility module re-exports keep pre-reorg paths such as
`euclid::length::LengthMeters` and `euclid::weights::TimeWeight` available even
though the implementation now lives under the layered tree. The crate-root
re-exports (`euclid::LengthMeters`, `euclid::TimeWeight`, ...) are the
recommended import path.

## Scalar API

Every `f64`-backed quantity implements `ScalarQuantity`, which provides finite
construction, ordering helpers, rounding, finiteness checks, and typed
approximate equality. Vector-like quantities also implement `LinearQuantity`,
which adds displacement operations such as `abs`, `signum`, `ratio`, `lerp`, and
`map`; absolute temperatures intentionally do not implement `LinearQuantity`.
`Interval<Q>` and affine `Point<Q>` are restricted to `LinearQuantity` so they
cannot accidentally perform vector arithmetic on absolute temperatures.

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
