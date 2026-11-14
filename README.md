# euclid

The base numeric and units package for Urban Fabric.

**Write a formula once; run it on anything that holds a number.**

A **carrier** is whatever actually holds the value — an ordinary `f64`, an
expression handle, a graph node, a tensor, a dataframe column, a unit-aware
wrapper. euclid is the vocabulary all of them share, so a formula written
against it compiles against every one without being rewritten.

```rust
use euclid_numeric::Real;

// Written once, against the trait — no mention of any concrete number type.
fn logpdf<T: Real>(x: T) -> T {
    -(T::half() * x.square()) - T::half_ln_tau()
}

let wide = logpdf(0.5_f64);    // eager host scalar
let narrow = logpdf(0.5_f32);  // a different carrier, same formula
```

Nothing in `logpdf` names `f64` or `f32`, so a carrier *you* define — an
expression handle to be compiled later, a graph node, a tensor element, a
length in metres — is accepted on the same bound with no edit to the formula.
That is the whole idea, applied to four kinds of value: scalars, physical
quantities, arrays, and randomness.

## Why This Exists

euclid is the numerics, units, and mathematical base for Urban Fabric. Its job
is to make a formula **usable across multiple interfaces** instead of being
rewritten for each one.

The same mathematics tends to be needed in several places at once — evaluated
eagerly on host scalars, staged into an expression or graph backend for
deferred execution, pushed down into dataframe expressions, or run elementwise
over an array runtime. Written the ordinary way, that means one implementation
per surface, and the copies drift: a coefficient gets corrected in one, a unit
conversion is right in the pipeline and wrong at the edge, a guard against
`log(0)` exists in two of four.

euclid removes the duplication by making the formula the single artifact and
the surface a parameter. `logpdf` above is written once; each interface
supplies a carrier, and the formula is then *the same code* everywhere it runs
— not a convention, not four implementations kept in sync by review.

The units half does the same thing one level up. Because a quantity is a trait
over a carrier rather than a concrete struct, every Urban Fabric project can
agree on what a length *is* — and on `length / time = speed` — without agreeing
on how a length is stored, or even whether it has been computed yet.

So the shared vocabulary is the deliverable: one definition of a number, one
definition of a physical quantity, reusable across every interface that needs
to do arithmetic.

## The Boundary

euclid's **contract** crates are contracts, not an engine. `euclid-units`
defines what a length *is* without shipping a `Meters` struct; `euclid-tensor`
defines the rank contract without shipping a tensor; `euclid-random` defines the
sampler contract without shipping a PRNG.

| the contract crates define | you implement |
|----------------|---------------|
| the operations a formula may use | storage layout and allocation |
| the trait tiers, plus `f64`/`f32`/integer/natural/`bool` impls | the execution runtime |
| physical dimensions and their cross-dimensional algebra | your own unit carriers |
| the array and sampler contracts | concrete tensors and a PRNG |

**Three crates in this repository are the other side of that table.** `oresme`
*is* the `Meters` struct. `napier` *is* the concrete `Copy` number line.
`frobenius` is linear-algebra content over the tensor contract. They live here
because each depends on nothing but euclid and nothing released them
independently — but they are kept as separate crates precisely so the boundary
above still holds: a consumer that takes `euclid-units` gets contracts only and
never sees `oresme`.

That distinction is the whole reason the crate list below is grouped rather
than flat.

The practical consequence is three rules that look arbitrary until you know
they exist to keep *deferred* carriers possible — you cannot branch on a graph
node, `Copy` a symbolic handle, or write a float literal into an expression
tree:

- comparisons return masks, never a host `bool`
- values are `Clone`, never `Copy`
- literals are lifted with `Real::scalar`, never written as `2.0`

## Crates

`euclid-numeric` is the foundation and has no dependencies at all. Everything
else depends on it and never on a peer, so a consumer takes on only the surface
it uses. **There is no umbrella crate** — depend on the specific crates you need.

```
euclid-numeric            no dependencies
├── euclid-units          → euclid-numeric
│   └── oresme            → euclid-units, euclid-numeric
├── euclid-tensor         → euclid-numeric
│   └── frobenius         → euclid-tensor, euclid-numeric
├── euclid-random         → euclid-numeric
└── napier                → euclid-numeric
```

**Contracts** — write a formula against these and it runs on any carrier.

| crate | purpose |
|-------|---------|
| [`euclid-numeric`](crates/euclid-numeric) | scalar algebra tiers, real analysis, masks, special functions, interpolation, graph control flow |
| [`euclid-units`](crates/euclid-units) | physical-dimension contracts over any `Real` carrier — SI base and derived quantities |
| [`euclid-tensor`](crates/euclid-tensor) | the rank-graded `Tensor` contract, its capability extensions, and dtype/spec metadata |
| [`euclid-random`](crates/euclid-random) | explicit-key random sampler contracts |

**Carriers and content** — the concrete side, folded in from their own
repositories because each depended on nothing but euclid.

| crate | purpose |
|-------|---------|
| [`oresme`](crates/oresme) | the concrete `f64` physical quantities — lengths, speeds, times, angles, byte sizes, counts, with uncertainty intervals and affine positions |
| [`napier`](crates/napier) | the concrete `Copy` number line (`Real`, `Complex`) plus the algebra tower and exact carriers (`Rational`, `Zn`, `GaussianRational`) |
| [`frobenius`](crates/frobenius) | matrix extensions beyond the rank contract — SPD, triangular solve, QR, eigen, SVD traits and the algorithms over them |

Only `oresme` pulls a third-party dependency (`bytemuck` and `serde`), and that
cost is per-crate: a consumer of the contract crates still gets none.

`euclid-utils` is a dev-only member — every crate takes it under
`[dev-dependencies]` for shared test helpers, and it is `publish = false`, so it
never reaches a consumer.

## Using It

```toml
[dependencies]
euclid-numeric = "0.3"   # the scalar foundation
euclid-units = "0.3"     # add only what you use
euclid-tensor = "0.3"
euclid-random = "0.3"

oresme = "0.3"           # concrete f64 quantities, if you need them
napier = "0.3"           # the concrete Copy number line
frobenius = "0.3"        # matrix extensions over the tensor contract
```

Crate names are hyphenated; in Rust they are imported with underscores:

```rust
use euclid_numeric::Real;
use euclid_units::Length;
```

Each crate's README covers its own trait surface. Beyond the three rules above,
one habit matters: **choose the narrowest trait bound that contains the
operations your formula needs** — `Ring` for `+ - *`, `Real` for
transcendentals, `Compare` only when you need masks or `select`. The bound is
the contract you are asking a carrier to satisfy, so a loose one rules out
carriers that could otherwise have run your code.

For the mask rule, the tools are `Compare::select`, `Compare::if_then_else`,
and the comparison-free `Real::min`/`max`/`clamp`. For the lifting rule, they
are `Real::scalar` and `Real::scalar_like`.

## Validation

Run the standard checks across the workspace before publishing changes:

```sh
cargo fmt --all --check
cargo test --all-targets --workspace
cargo clippy --all-targets --workspace -- -D warnings
cargo test --doc --workspace
```

Lints (`unsafe_code`, `unreachable_pub`, `unwrap_used`, `expect_used`, `panic`,
`self_named_module_files`) are declared once in `[workspace.lints]` at the root
and inherited by each crate. Shared package metadata comes from
`[workspace.package]`, so all four crates version together.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2026 Raymond Mead.

Apache-2.0 governs what a recipient may do with the code; it does not oblige
anyone to publish it. Keeping this repository private is a separate, purely
practical matter. Note the corollary: anyone who *does* receive it under
Apache-2.0 holds an irrevocable right to redistribute it, publicly included.
