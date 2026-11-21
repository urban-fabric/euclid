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
use euclid_scalar::Real;

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

`euclid-algebra` supplies the algebraic structures and exact number systems. `euclid-scalar`
combines scalar contracts, native implementations and numerical kernels. `euclid-quantity`
supplies concrete physical quantities, while `euclid-matrix` extends the tensor contract.

### Two scalar tiers in one crate

The portable scalar traits allow deferred carriers: values are `Clone`, literals are lifted
through `Real::scalar`, and comparisons return masks. `HostReal` and `HostComplex` add the
requirements of eager scalar code: `Copy`, associated constants and host predicates. These
are separate traits within `euclid-scalar`, not separate crates.

| use | trait |
|---|---|
| backend-neutral real formulas, including expressions and tensors | `euclid_scalar::Real` |
| eager real kernels requiring `Copy` and host predicates | `euclid_scalar::HostReal` |
| backend-neutral complex construction and components | `euclid_scalar::Complex` |
| eager complex field and transcendental operations | `euclid_scalar::HostComplex` |

The native `f64`/`f32`, integer and boolean implementations live beside their traits. The
special-function algorithms live in private `kernels/` modules, and the public trait methods
delegate to them. A new backend implements the contracts; it does not inherit a storage layout.

## Crates

**There is no umbrella crate** — depend on the crates you use. The production graph has four
layers; the tree shows dependencies from foundation to consumer:

```text
euclid-algebra              algebraic structures and exact carriers
└── euclid-scalar            scalar contracts, host traits, primitive impls, numerical kernels
    ├── euclid-units         dimension and quantity contracts
    │   └── euclid-quantity  concrete physical quantities; also uses scalar
    ├── euclid-tensor        rank-graded tensor contracts
    │   └── euclid-matrix    matrix extensions; also uses scalar
    └── euclid-random        sampler contracts

Test tooling, outside the production layers:
  euclid-stability          property runner, shrinking and numerical instruments
  euclid-utils              test helpers
```

`euclid-scalar` depends on `euclid-algebra`. The three domain contracts depend on `euclid-scalar`
and not on one another. Quantity and matrix extend their corresponding domain contracts, and
also use scalar contracts directly. Optional `conformance` features add the independent stability
engine explicitly; ordinary consumers do not receive the test tooling.

The nine workspace members include seven production crates and two test-tooling crates.
`euclid-numeric` was merged into `euclid-scalar` for 0.6: scalar traits and scalar implementations
now have one owner. See [the migration notes](RELEASES.md#060).

| concern | contract | implementation |
|---|---|---|
| scalar mathematics | [`euclid-scalar`](crates/euclid-scalar) | primitive implementations in the same crate |
| physical quantities | [`euclid-units`](crates/euclid-units) | [`euclid-quantity`](crates/euclid-quantity) |
| tensors | [`euclid-tensor`](crates/euclid-tensor) | supplied by a backend |
| randomness | [`euclid-random`](crates/euclid-random) | supplied by a consumer |
| algebraic structure | [`euclid-algebra`](crates/euclid-algebra) | exact carriers in the same crate |
| matrix extensions | [`euclid-matrix`](crates/euclid-matrix) | supplied by a backend |

**Contracts** — formulas written against these run on any suitable carrier. None introduces
third-party production dependencies. `euclid-quantity` additionally uses serde and bytemuck for
its concrete storage wrappers.

A contract crate is also where you find out whether your carrier *honors* it. A
tier is a claim — `impl Semiring for MyThing {}` asserts that `⊗` distributes
over `⊕`, and the type system checks neither — so the laws ship as public
functions you call against a sample of your own type:

```rust,ignore
euclid_algebra::conformance::semiring_laws(&my_sample, "MyThing");
euclid_algebra::conformance::ring_laws(&my_sample, "MyThing");
```

They sit behind a `conformance` feature, so a consumer that only writes formulas
pays nothing for them. **All six contract crates ship them**, and `--all-features`
is part of the documented check set for that reason — the gated code is not
compiled otherwise, so a broken law would pass every other check.

`euclid-matrix` is the one whose suites would have to be *run elsewhere*: its
traits are backend primitives, so only a concrete tensor can satisfy them, and
euclid ships none.

### The laws come in two strategies

Each law takes a slice, and there are two ways to get one. Both are public, and
running both is the recommendation:

```rust,ignore
// The fixed sample you declare — six values you chose because they stress your carrier.
euclid_tensor::conformance::tensor_laws_for::<MyTensor>("MyTensor");

// The generated search euclid ships — shapes weighted toward the cases that hide bugs.
euclid_tensor::conformance::generated::tensor_laws_generated::<MyTensor>("MyTensor");
```

They answer different questions. A **fixed** sample proves a tier is
*satisfiable* and that your carrier satisfies it, and a failure is a value in
your source with no seed to recover. A **generated** one hunts the thing a fixed
list cannot cover — your carrier's *representation*, which is a large state
space rather than a structural claim.

The second used to live in euclid's private test tree, which meant euclid shipped
half its testing strategy and every backend rediscovered the other half. What
was being withheld is specific and worth naming:

- `euclid-tensor` weights extents toward **repeated** ones, where a permutation
  applied backwards produces the right shape and the wrong data, and reaches
  **rank 4**. euclid's own `structural_laws` was malformed at rank ≥ 4 for as
  long as its reference sample stopped at rank 3 — a backend whose sample stops
  where euclid's did inherits exactly that blind spot.
- `euclid-random` **deduplicates its seeds by construction**, because `key_laws`
  asserts every *pair* draws differently and a repeated seed therefore fails it
  for reasons that say nothing about the backend.
- `euclid-units` draws signs and half-integers, because the defaulted `round`,
  `trunc`, `floor` and `clamp` bodies branch on orderings and signs rather than
  on structure.

Two crates needed nothing new to export this — `Tensor::from_host` and
`Measure::from_raw` already let euclid build a carrier. `euclid-algebra` and
`euclid-scalar` do need one thing from you, a `GeneratedSample` impl, because a
`Semiring` has no constructor and should not: "build me one from a number" is not
part of being one.

The engine comes with the feature. `euclid-stability` is an optional dependency
of the five contract crates, enabled by `conformance` — so a consumer that only
binds the contract never sees it, and one that asks for the test surface gets
what the test surface is written against.

| crate | purpose |
|-------|---------|
| [`euclid-scalar`](crates/euclid-scalar) | scalar traits, host traits, primitive implementations, numerical kernels, constants and conformance laws |
| [`euclid-units`](crates/euclid-units) | physical-dimension contracts over any `Real` carrier — SI base and derived quantities |
| [`euclid-tensor`](crates/euclid-tensor) | the rank-graded `Tensor` contract, its capability extensions, and dtype/spec metadata |
| [`euclid-random`](crates/euclid-random) | explicit-key random sampler contracts |
| [`euclid-matrix`](crates/euclid-matrix) | matrix extensions beyond the rank contract — the SPD, triangular-solve, QR, eigen and SVD signatures a backend implements |
| [`euclid-algebra`](crates/euclid-algebra) | the abstract-algebra tower — the semiring/dioid trunk, the ring/field branch, the semimodule action, the witnessed path algebras (min-plus, max-plus, widest-path), and the exact carriers (`Rational`, `Zn`/`Zp`, `GaussianRational`) |

**Carriers** — the concrete side, which breaks the three rules on purpose.

| crate | purpose |
|-------|---------|
| [`euclid-quantity`](crates/euclid-quantity) | the concrete `f64` physical quantities — lengths, speeds, times, angles, byte sizes, counts, with uncertainty intervals and affine positions |

**Dev-only** — two further members, both `publish = false`, neither of which a
consumer receives unless it asks. `euclid-utils` is taken only under
`[dev-dependencies]` and so appears nowhere in the graph above.
`euclid-stability` is **also an optional runtime dependency of the five contract
crates, enabled by their `conformance` feature** — see
[the generated laws](#the-laws-come-in-two-strategies) for why the engine has to
travel with them.

| crate | purpose |
|-------|---------|
| [`euclid-utils`](crates/euclid-utils) | shared test helpers — comparison criteria, finite-difference calculus, domain sweeps, shape assertions |
| [`euclid-stability`](crates/euclid-stability) | numerical stability and property testing — condition numbers, forward-error bounds, seam hunting, accuracy budgets, and randomized search with shrinking |

`euclid-stability` is the one member that looks like it fails the [membership
rule](#the-membership-rule) and does not. A test engine is not something the
fleet must agree on numerically — but it is also not a *deliverable*: it exists
so the suites in this repository can search for the input that breaks a contract
rather than asserting a fixed list of them. It replaces `proptest`, which
`deny.toml` bans for a stated reason: the fleet's generators are mostly
*dependent* (a rank that bounds the axes indexing it, a node count that bounds
the edges after it), and that is proptest's documented weak spot — nested
`prop_flat_map` shrinking is capped and silently stops improving the
counterexample. Generating from a tape re-derives a dependent bound on every
candidate instead, so there are no shrink implementations to write and adding a
generator cannot introduce a shrinking bug.

Each contract crate's conformance laws are now run twice: once over a fixed
sample, and once over generated input. The two answer different questions, which
[CONTRIBUTING](CONTRIBUTING.md#what-a-change-should-come-with) spells out — the
fixed sample proves a tier is satisfiable, the generated one hunts the
representation invariants underneath it.

**The name says "stability" rather than "property" on purpose.** It was
`euclid-property` until 0.5. QuickCheck associates generation and per-type
shrinking through `Arbitrary`; this crate instead edits a replayable choice
tape. It has no `Arbitrary`, deliberately — a generator is
declared, in the same spirit as `ConformanceSample` — and its weight is
numerical rather than type-theoretic: condition numbers, forward and two-stage
error bounds, seam hunting at algorithm crossovers, ulp monotonicity scans, and
accuracy budgets. The property-testing machinery is the foundation those sit on
rather than the point, which is why they are one crate.

The budget is the piece worth knowing about from outside, because it closes a
gap every other check in this repository leaves open:

```rust,ignore
Config::default().budget("tests/budgets.txt")   // alongside target(observed_error)
```

A property asserts a contract — "within 8 ulp" — with enough headroom to admit
today's error and tomorrow's. So a refactor that makes `erf` twice as inaccurate
passes every check here: the property still holds. `target` already hunts the
worst case rather than sampling uniformly; `budget` writes that worst score to a
committed file and **fails a later run that scores higher**. The assertion stays
the contract; the baseline notices the day the contract starts being used up.
It works because the search seed is fixed, so the number is a property of the
code rather than of the run.

### Which kernels euclid holds

This is the [membership rule](#the-membership-rule) applied to the hardest case,
where "is this a contract?" gives the wrong answer and "must this be agreed?"
gives the right one:

- **Scalar kernels belong in euclid.** `erf`, `lgamma`, the Bessel and Airy
  functions and the rest of `euclid-scalar/src/kernels/` are ~8.4k lines of
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

### The line is the data structure

The membership rule says *whether* something must be agreed. What decides when
agreement is cheap enough to ship an implementation is simpler than it looks:

> **euclid implements over universal datatypes. It never ships a data
> structure.**

`f64` is universal. Every platform has it, it means the same thing everywhere,
and there is nothing to design — so `erf(x: f64) -> f64` can be written once
here and be the answer for everyone. The same goes for `f32`, the integers, and
`bool`, and for a newtype over one of them: `LengthMeters(f64)` and `Zp<P>(u64)`
add a name and a dimension, not a structure.

A tensor, a matrix or a graph is different in kind. None of them can be
implemented without *designing a data structure* first — dense or sparse, row or
column major, owned or borrowed, host or device, CSR or COO — and every one of
those is a real choice with real trade-offs. They are not mistakes to be
eliminated by picking one; they are the reason backends exist. So euclid names
the operations and a backend brings the structure.

That is the whole explanation for a boundary that otherwise looks arbitrary:

| | euclid ships | because |
|---|---|---|
| `erf`, `lgamma`, Bessel, Airy | the implementation | `f64` is universal; one right answer |
| `Real`, `Complex`, `Rational`, `Zp` | the implementation | newtypes over universal datatypes |
| `LengthMeters`, `TimeSeconds`, … | the implementation | same, with a dimension attached |
| `Tensor`, `SpdSolve`, `QrDecomposition` | the signature only | needs a data structure, which is a design |
| a PRNG | the signature only | needs state, and shaped output needs a collection |

The dimensionality of a value is the symptom rather than the cause. A scalar
needs no data structure, which is why euclid can implement over it; a collection
needs one, which is why euclid cannot. The reference tensor in this repository,
`Dense`, lives in a private test module for exactly that reason — it is a data
structure, so it is scaffolding rather than a deliverable.

So `euclid-quantity` being 14k lines of concrete `f64` is not a tension with the
rest of the repository. Every one of those types is a newtype over a universal
datatype, which is the same thing the scalar kernels are built on — a metre is
an `f64` with a name.

`euclid-matrix` used to sit across that line, holding ~126 lines of matrix
kernels (`least_squares`, `gram`, `inverse`, `dominant_eigenvector`, …) beside
its signatures. Those are gone; it is now vocabulary only, and a contract crate
like the rest. An algorithm over those traits lives with a backend, so two
backends wanting the same one write it twice — that is the cost of the boundary,
and the right trade, since a shared algorithm that cannot be tested without
euclid shipping a tensor is worse than two tested ones.

Only `euclid-quantity` pulls a third-party dependency (`bytemuck` and `serde`),
and that cost is per-crate: a consumer of the contract crates still gets none.

`euclid-utils` and `euclid-stability` are the two dev-only members, both
`publish = false`. `euclid-utils` is purely a `[dev-dependencies]` entry and
reaches no consumer at all. `euclid-stability` reaches one that enables a
`conformance` feature, and only then — the generated laws are written against
it, so the engine travels with the test surface a consumer opted into.

## Using It

**These crates are not on crates.io.** The current release is 0.7.0. Against a local
checkout, use paths:

```toml
[dependencies]
euclid-scalar = { version = "0.7", path = "../euclid/crates/euclid-scalar" }
euclid-units = { version = "0.7", path = "../euclid/crates/euclid-units" }
euclid-tensor = { version = "0.7", path = "../euclid/crates/euclid-tensor" }
euclid-random = { version = "0.7", path = "../euclid/crates/euclid-random" }
euclid-algebra = { version = "0.7", path = "../euclid/crates/euclid-algebra" }
euclid-quantity = { version = "0.7", path = "../euclid/crates/euclid-quantity" }
euclid-matrix = { version = "0.7", path = "../euclid/crates/euclid-matrix" }
```

Note that `euclid-quantity` no longer re-exports `euclid-units`, so take `euclid-units`
directly if you name `Quantity`, `Length` or any other unit marker.

For released versions, depend on this private repository by git and **pin its actual release
tag**, rather than a branch. `v0.5.0` retains the old crate split; use
[the migration notes](RELEASES.md#070) when moving from 0.6.0.

Take only the crates you use; there is no umbrella crate to pull the rest in.
Every member is Apache-2.0 — see [License](#license).

### Why a tag and not `branch = "main"`

This README told you to track `main` until 0.5, and that was the single thing
standing between euclid and being depended on safely. A branch dependency has no
version: cargo re-resolves it to whatever `main` points at, so **every consumer
silently tracked HEAD**. A trait gaining a required method, a crate being
renamed, a conformance law being corrected — each arrived unannounced in the
next `cargo update`, and euclid's whole purpose is that independently-written
parts of the fleet agree on definitions. Agreeing on a moving definition is not
agreement.

A tag is immutable, so the version you compiled against is the version you keep
until you choose otherwise. The `version` field beside it is not redundant:
cargo checks it against the manifest at that ref, so a tag pointing somewhere
unexpected fails resolution instead of compiling something you did not ask for.

**What breaks between versions is recorded in [RELEASES](RELEASES.md)** — only
the breaking changes, because that is the only question a pinned consumer has.
The minor number is the breaking axis: `0.6.x` is compatible, `0.7.0` may not
be.

### One ref per build graph

Pin the same Euclid tag throughout a dependency graph. Cargo can resolve
multiple Git refs in one build, where each ref supplies distinct trait and type
identities. A quantity or carrier from one ref cannot satisfy the corresponding
type or trait from another. Upgrade the dependencies that expose Euclid types
first, and check for duplicate Euclid packages with `cargo tree --duplicates`.

### euclid is layer 1

"The bottom" is not a figure of speech. The fleet's target architecture is
**stratified**, and euclid is the floor of it:

| layer | repositories | may depend on |
|---|---|---|
| **1** | **euclid** | nothing in the fleet |
| 2 | `bayes`, `hausdorff`, `cayley` | layer 1 |
| 3 | `abel` | layers 1–2 |
| 4 | `sierpinski`, `galois` | layers 1–3 |

A layer may depend on any layer *below* it, never on its own or above. That is
what makes the graph acyclic by construction and an upgrade a finite chain: a
euclid release propagates to layer 2, then 3, then 4, and each layer can move
only once everything beneath it has.

**What this obliges of euclid**, and the reason it is written here rather than
somewhere upstream: layer 1 may depend on nothing in the fleet, so euclid stays
dependency-free apart from `bytemuck` and `serde` in `euclid-quantity`. Any edge
out of this repository into a sibling would make the floor depend on something
built on it. There is none, and `cargo deny`'s `[bans]` section is what keeps it
that way.

It also sets euclid's release obligation. Being the floor means a breaking change
here is the most expensive kind in the fleet — it is the only one every other
repository must absorb — which is why [RELEASES](RELEASES.md) records breaking
changes and nothing else, and why the minor number is the axis.

#### Checked, as of 0.5.0

A target architecture nobody has verified is a diagram, so this one was checked
against the manifests rather than drawn. The edges, by repository:

```text
euclid       -> (none)
bayes        -> euclid
hausdorff    -> euclid
cayley       -> euclid
abel         -> euclid, bayes, cayley, hausdorff
sierpinski   -> euclid
galois       -> euclid, bayes, cayley, hausdorff, abel
```

**No violations**: every edge points strictly downward. `abel` is layer 3 rather
than a peer of the three it depends on, and `galois` is layer 4 above it, which
is what makes that true.

Note `sierpinski` reaches only euclid, so nothing forces it as high as layer 4 —
a layer is a ceiling on what you may depend on, not a requirement to use it, and
it is placed where it is for reasons the dependency graph does not record.

This is a point-in-time check against sibling working copies, not something the
test suite can enforce: layer 1 may depend on nothing in the fleet, so euclid
cannot see the fleet in order to verify it. Re-check it when the shape changes.

#### Three repositories above layer 4

`cantor`, `weierstrass` and `dirichlet` are not in the table, and their existing
edges force where they land:

```text
dirichlet    -> euclid, sierpinski(4)                           >= layer 5
weierstrass  -> galois(4)                                       >= layer 5
cantor       -> hausdorff(2), sierpinski(4), galois(4), dirichlet(5)  >= layer 6
```

So the minimal consistent assignment is layer 5 for `dirichlet` and
`weierstrass`, layer 6 for `cantor` — which incidentally resolves the one
remaining same-layer edge, `cantor -> dirichlet`, by separating them. Those three
are recorded as *derived* rather than decided; confirm or move them.

**The deepest chain is therefore six layers**, and that number is the one that
matters here: it is how many sequential repository migrations a breaking euclid
release costs.

Crate names are hyphenated; in Rust they are imported with underscores:

```rust
use euclid_scalar::Real;
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
cargo test --all-targets --workspace --all-features
cargo clippy --all-targets --workspace --all-features -- -D warnings
cargo test --doc --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items --all-features
cargo deny check
```

All six, and [CONTRIBUTING](CONTRIBUTING.md#checks-run-locally-not-in-ci) says why each one is
there — in particular why `--all-features` is not padding (the `conformance`
suites are feature-gated, so the default build never compiles them) and what
`cargo deny check` is enforcing beyond vulnerabilities.

Lints (`unsafe_code`, `unreachable_pub`, `unwrap_used`, `expect_used`, `panic`,
`mod_module_files`) are declared once in `[workspace.lints]` at the root and
inherited by each crate — including `euclid-algebra`, `euclid-scalar` and
`euclid-matrix`, which arrived from their own repositories with a warn-level
copy of the same table. Shared package metadata comes from
`[workspace.package]`, so all ten crates version together and under one licence
— including those same three, which carried `license = "Proprietary"` until 0.4.
See [License](#license) for what `cargo deny` made of that.

`mod_module_files` means a module with children is `foo.rs` beside `foo/`, never
`foo/mod.rs`. Note the near-miss: `self_named_module_files` is the *opposite*
lint and demands `mod.rs`.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2026 Raymond Mead.

**There are no exceptions.** Every one of the nine members takes
`license.workspace = true`, so the whole workspace is Apache-2.0 and versions
together under one licence.

Three of them did not, until 0.4. `euclid-algebra` and `euclid-matrix` arrived
from their own repositories and `euclid-scalar` was split out of
`euclid-algebra`, and each carried `license = "Proprietary"` with a comment
arguing that a move should not relicense code and neither should a split. That
has been overridden deliberately by the copyright holder: the fleet ships under
one licence, and a single member under different terms makes the workspace
unpublishable as a set.

`cargo deny check` is what surfaced it, and the detail is worth keeping.
`"Proprietary"` is not a valid SPDX expression, so the three crates did not read
as *restrictively licensed* — they read as **unlicensed**, which is a worse
position than either, and no amount of care in the comment above the field
changed what a tool would conclude.

Apache-2.0 governs what a recipient may do with the code; it does not oblige
anyone to publish it. Keeping this repository private is a separate, purely
practical matter. Note the corollary, which is now uniform across the
workspace: anyone who *does* receive a copy under Apache-2.0 holds an
irrevocable right to redistribute it, publicly included.
