# euclid

**euclid is where Urban Fabric agrees with itself about numbers.**

It holds the definitions that independently-written parts of the fleet have to
mean *identically* — what a real number supports, what a length is, which
algebraic tier a carrier reaches, what `erf(0.7)` returns. Anything those parts
may legitimately differ on lives with them instead.

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

### The problem is silent disagreement

Thirty-odd repositories compute over the same quantities. `abel` works out a
drive time, `ramsey` a betweenness estimate, `galois` pushes the same formula
into a dataframe, a backend runs it elementwise over an array. Nothing makes
them agree on what a length is, on what `+` means at which tier, or on what
`erf` returns at 0.7.

When they drift, **nothing fails.** There is no type error and no panic — you
get two different numbers from two places that were supposed to be computing
the same thing, and no indication which is wrong. That is the failure euclid
exists to prevent, and it is why the deliverable is a *definition* rather than
a library of helpers.

### The membership rule

> **If two independently-written parts of the fleet must agree on it, it belongs
> in euclid. If they may legitimately differ, it belongs with them.**

That is the whole test, and it is meant to be usable on a single file without
appeal to taste:

| | must agree? | lives |
|---|---|---|
| `Real`, `Length`, `Tensor`, `Semiring` | yes — otherwise formulas do not compose | here |
| `erf`, `lgamma`, Bessel, Airy | yes — one right answer per input | here |
| `LengthMeters`, `Rational`, `Zp`, min-plus | yes — otherwise two projects cannot exchange a value | here |
| Cholesky, QR, SVD | **no** — blocked, packed or GPU is the backend's call | a backend |
| a PRNG algorithm | no — backends differ for good reasons | a backend |
| CSR layout, traversal order, route costs | no | `cayley`, `abel` |

Note what the rule does *not* say. It is not "traits, not implementation" —
`erf` is 8,400 lines of concrete `f64` code and belongs here, because there is
one right answer and a second implementation is a second chance to be wrong.
`cholesky` is also implementation and does *not* belong here, because it has
many valid implementations and choosing between them is exactly the backend's
job. **Both are implementation; only one requires agreement.**

The practical consequence is that euclid is mostly trait surface, but not as a
rule — as a result. Most things that must be agreed are best expressed as a
contract, and the exceptions are admitted on their merits rather than smuggled
in.

### What agreement buys

Once the vocabulary is shared, a formula written against it is **usable across
every interface** instead of being rewritten for each one.

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

A fifth contract crate, `euclid-algebra`, sits beside those: it defines what
`⊕` and `⊗` must *satisfy* — which algebraic tier a carrier reaches — and ships
nothing concrete beyond a few exact number systems.

**Three crates in this repository are the other side of that table.**
`euclid-quantity` *is* the `Meters` struct and `euclid-scalar` *is* the concrete
`Copy` number line; `euclid-matrix` is linear-algebra *content* — algorithms
written once over the tensor contract, naming no backend. They live here because
each depends on nothing but euclid and nothing released them independently — but
they are kept as separate crates precisely so the boundary above still holds: a
consumer that takes `euclid-units` gets contracts only and never sees
`euclid-quantity`.

That distinction is the whole reason the crate list below is grouped rather
than flat.

### Three rules, and who they bind

The **contract** crates follow three rules that look arbitrary until you know
they exist to keep *deferred* carriers possible — you cannot branch on a graph
node, `Copy` a symbolic handle, or write a float literal into an expression
tree:

- comparisons return masks, never a host `bool`
- values are `Clone`, never `Copy`
- literals are lifted with `Real::scalar`, never written as `2.0`

**The carrier crates break all three, deliberately.** That is not drift, it is
their purpose. `euclid-scalar`'s `Real` is `Copy`, returns a host `bool` from
`is_finite`, and offers `PI` as an associated constant rather than a lifted
literal — because eager geometry and physics kernels genuinely need the real
number line, and a tensor compiler cannot give them one. Every `euclid-quantity`
unit is likewise `Copy` with `bool` predicates.

So the rules are a property of the *tier*, not of the repository:

| | contract tier | carrier tier |
|---|---|---|
| crates | `euclid-numeric`, `euclid-units`, `euclid-tensor`, `euclid-random`, `euclid-algebra` | `euclid-quantity`, `euclid-scalar` |
| comparison yields | a carrier mask | a host `bool` |
| value semantics | `Clone` | `Copy` |
| constants | lifted via `Real::scalar` | associated constants |
| runs on | any carrier, including deferred ones | concrete `f64`/`f32` only |

Write against the contract tier by default: it is the wider bound, and it is
what makes a formula portable. Reach for the carrier tier only when a kernel
needs the number line itself.

## Crates

There are **two dependency-free roots**, not one. `euclid-numeric` is the larger
and the one most things reach for, but `euclid-algebra` is not a leaf of it:
algebraic structure (which tier a carrier reaches) needs nothing from analysis
(what operations a formula may use). The two foundations sit side by side and
meet only in `euclid-scalar`, the one crate with two parents — which is exactly
what that crate is for.

**There is no umbrella crate** — depend on the specific crates you need.

```
euclid-numeric             no dependencies
├── euclid-units           → euclid-numeric
│   └── euclid-quantity    → euclid-units, euclid-numeric
├── euclid-tensor          → euclid-numeric
│   └── euclid-matrix      → euclid-tensor, euclid-numeric
├── euclid-random          → euclid-numeric
└── euclid-scalar          → euclid-numeric, euclid-algebra
                                              ↑
euclid-algebra             no dependencies ───┘
```

Three crates depend on a **peer** as well as on a root — `euclid-quantity` on
`euclid-units`, `euclid-matrix` on `euclid-tensor`, `euclid-scalar` on
`euclid-algebra`. In every case the peer is the contract the crate implements or
extends, so the edge is the point rather than a shortcut. A consumer still takes
on only the surface it uses: nothing depends *downward* from a root onto a
carrier, so taking `euclid-units` never pulls `euclid-quantity`.

The eight published members fall out of two questions: **which concern** is this,
and **contract or concrete**. Drawn as a grid, the crate list stops looking
arbitrary — and the empty cells are as deliberate as the full ones. The first
four rows are the four kinds of value from the top of this page:

| concern | contract | concrete |
|---|---|---|
| scalars | [`euclid-numeric`](crates/euclid-numeric) | [`euclid-scalar`](crates/euclid-scalar) |
| physical quantities | [`euclid-units`](crates/euclid-units) | [`euclid-quantity`](crates/euclid-quantity) |
| arrays | [`euclid-tensor`](crates/euclid-tensor) | — *a backend supplies one* |
| randomness | [`euclid-random`](crates/euclid-random) | — *a consumer supplies a PRNG* |
| algebraic structure | [`euclid-algebra`](crates/euclid-algebra) | — |
| matrix extensions | [`euclid-matrix`](crates/euclid-matrix) | — *a backend supplies one* |

**Contracts** — write a formula against these and it runs on any carrier. All
six are dependency-free in the sense that matters: nothing third-party.

A contract crate is also where you find out whether your carrier *honors* it. A
tier is a claim — `impl Semiring for MyThing {}` asserts that `⊗` distributes
over `⊕`, and the type system checks neither — so the laws ship as public
functions you call against a sample of your own type:

```rust
euclid_algebra::conformance::semiring_laws(&my_sample, "MyThing");
euclid_algebra::conformance::ring_laws(&my_sample, "MyThing");
```

They sit behind a `conformance` feature, so a consumer that only writes formulas
pays nothing for them. `euclid-algebra` has them today; the other contract
crates are expected to follow.

| crate | purpose |
|-------|---------|
| [`euclid-numeric`](crates/euclid-numeric) | scalar algebra tiers, real analysis, masks, special functions, interpolation, graph control flow |
| [`euclid-units`](crates/euclid-units) | physical-dimension contracts over any `Real` carrier — SI base and derived quantities |
| [`euclid-tensor`](crates/euclid-tensor) | the rank-graded `Tensor` contract, its capability extensions, and dtype/spec metadata |
| [`euclid-random`](crates/euclid-random) | explicit-key random sampler contracts |
| [`euclid-matrix`](crates/euclid-matrix) | matrix extensions beyond the rank contract — the SPD, triangular-solve, QR, eigen and SVD signatures a backend implements |
| [`euclid-algebra`](crates/euclid-algebra) | the abstract-algebra tower — the semiring/dioid trunk, the ring/field branch, the semimodule action, the witnessed path algebras (min-plus, max-plus, widest-path), and the exact carriers (`Rational`, `Zn`/`Zp`, `GaussianRational`) |

**Carriers** — the concrete side, which breaks the three rules on purpose.

| crate | purpose |
|-------|---------|
| [`euclid-quantity`](crates/euclid-quantity) | the concrete `f64` physical quantities — lengths, speeds, times, angles, byte sizes, counts, with uncertainty intervals and affine positions |
| [`euclid-scalar`](crates/euclid-scalar) | the concrete `Copy` number line — `Real` and `Complex`, where the algebra tower and the analysis surface fuse |

### Which kernels euclid holds

This is the [membership rule](#the-membership-rule) applied to the hardest case,
where "is this a contract?" gives the wrong answer and "must this be agreed?"
gives the right one:

- **Scalar kernels belong in euclid.** `erf`, `lgamma`, the Bessel and Airy
  functions and the rest of `euclid-numeric/src/kernels/` are ~8.4k lines of
  concrete `f64`/`f32` code, and they stay. A scalar special function has one
  right answer, every carrier needs it, and a second implementation of `erf` in
  a consumer is a second chance to get it wrong. It is also what makes
  `f64: Real` satisfiable at all — the reference carrier has to work.
- **Matrix and tensor kernels do not.** `cholesky` has many valid
  implementations — blocked, packed, GPU — and choosing between them is exactly
  what a backend is for, so there is nothing to agree on. `euclid-tensor` and
  `euclid-random` therefore define their contracts and ship no carrier; a
  backend supplies the tensor and the PRNG.

Both are implementation. Only one requires agreement, which is why the rule is
about agreement and not about traits.

`euclid-matrix` used to sit across that line, holding ~126 lines of matrix
kernels (`least_squares`, `gram`, `inverse`, `dominant_eigenvector`, …) beside
its signatures. Those are gone; it is now vocabulary only, and a contract crate
like the rest. An algorithm over those traits lives with a backend, so two
backends wanting the same one write it twice — that is the cost of the boundary,
and the right trade, since a shared algorithm that cannot be tested without
euclid shipping a tensor is worse than two tested ones.

Only `euclid-quantity` pulls a third-party dependency (`bytemuck` and `serde`),
and that cost is per-crate: a consumer of the contract crates still gets none.

`euclid-utils` is a dev-only ninth member — the contract crates take it under
`[dev-dependencies]` for shared test helpers, and it is `publish = false`, so it
never reaches a consumer.

## Using It

**These crates are not on crates.io.** This repository is private, so depend on
it by git — the version numbers below describe the tier, not a registry release:

```toml
[dependencies]
# the contract tier — write formulas against these
euclid-numeric = { git = "ssh://git@github.com/urban-fabric/euclid.git", branch = "main" }
euclid-units   = { git = "ssh://git@github.com/urban-fabric/euclid.git", branch = "main" }
euclid-tensor  = { git = "ssh://git@github.com/urban-fabric/euclid.git", branch = "main" }
euclid-random  = { git = "ssh://git@github.com/urban-fabric/euclid.git", branch = "main" }
euclid-algebra = { git = "ssh://git@github.com/urban-fabric/euclid.git", branch = "main" }

# the carrier tier — concrete f64, only if a kernel needs the number line
euclid-quantity = { git = "ssh://git@github.com/urban-fabric/euclid.git", branch = "main" }
euclid-scalar   = { git = "ssh://git@github.com/urban-fabric/euclid.git", branch = "main" }

# content — algorithms over the tensor contract
euclid-matrix = { git = "ssh://git@github.com/urban-fabric/euclid.git", branch = "main" }
```

Take only the crates you use; there is no umbrella crate to pull the rest in.
Note that `euclid-algebra`, `euclid-scalar` and `euclid-matrix` are licensed
differently from the others — see [License](#license).

Crate names are hyphenated; in Rust they are imported with underscores:

```rust
use euclid_numeric::Real;
use euclid_units::Length;
```

Every published crate's README covers its own trait surface (`euclid-utils` is
the dev-only member and has none). Beyond the three rules above,
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
`mod_module_files`) are declared once in `[workspace.lints]` at the root and
inherited by each crate — including `euclid-algebra`, `euclid-scalar` and
`euclid-matrix`, which arrived from their own repositories with a warn-level
copy of the same table. Shared package
metadata comes from `[workspace.package]`, so all nine crates version together.

`mod_module_files` means a module with children is `foo.rs` beside `foo/`, never
`foo/mod.rs`. Note the near-miss: `self_named_module_files` is the *opposite*
lint and demands `mod.rs`.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2026 Raymond Mead.

**Three crates are excepted.** `euclid-algebra`, `euclid-scalar` and
`euclid-matrix` each declare `license = "Proprietary"` rather than inheriting
the workspace's Apache-2.0. `euclid-algebra` and `euclid-matrix` arrived here
from their own repositories, and `euclid-scalar` was split out of
`euclid-algebra` — a move should not relicense code, and neither should a split.
Everything else — the four contract crates, `euclid-quantity`, and the dev-only
`euclid-utils` — takes `license.workspace = true` and is Apache-2.0.

Apache-2.0 governs what a recipient may do with the code; it does not oblige
anyone to publish it. Keeping this repository private is a separate, purely
practical matter. Note the corollary: anyone who *does* receive it under
Apache-2.0 holds an irrevocable right to redistribute it, publicly included —
which is exactly why the two proprietary crates are tracked separately above.
