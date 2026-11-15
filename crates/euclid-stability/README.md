# euclid-stability

Numerical stability and property testing. Part of [euclid](../..).

Dev-only (`publish = false`), on the same footing as `euclid-utils`: take it under
`[dev-dependencies]`. These crates are not on crates.io — see
[the root README](../../README.md#using-it) for the git dependency to add.

```rust
use euclid_stability::{check_eq, generate, property};

property! {
    fn reversing_twice_is_the_identity(xs in generate::vec(generate::int(0, 9), 0, 8)) {
        let mut round_trip = xs.clone();
        round_trip.reverse();
        round_trip.reverse();
        check_eq!(round_trip, xs);
    }
}
```

## The idea: a replayable choice tape

A generator never touches a random number generator. It draws from a `Source`, which **records
every choice onto a tape** of `u64`s. A generated value is therefore a pure function of its
tape.

That one property turns shrinking from a per-type problem into a generic one. Instead of every
type explaining how to make itself smaller, the shrinker edits the *tape* — deleting runs,
lowering entries, binary-searching magnitudes — and re-runs the generator. Whatever comes out is
structurally valid by construction, because it came from the generator.

**There are no shrink implementations to write.** A new generator composes out of `generate` and
is immediately as shrinkable as everything else. Adding a generator cannot introduce a shrinking
bug.

## Why not proptest

Not supply chain and not licensing — proptest is MIT/Apache-2.0 and perfectly reputable. The
reason is **dependent generators**, which is what the fleet mostly writes: a node count that
bounds the edge endpoints after it, a matrix dimension that bounds the indices into it.

That is proptest's documented weak spot. From its own source, on `prop_flat_map`:

> As a result, nested `prop_flat_map`/`Flatten` combinators risk exponential run time on this
> search for new failing values. To ensure that test failures occur within a reasonable amount
> of time, all of these combinators share a single "flat map regen" counter, and will stop
> generating new values if it exceeds `Config::max_flat_map_regens`.

A tape has no such limit. Lowering the entry that produced the node count means the generator
re-reads the following entries under the *new* bound, so the dependency is re-derived for free
and every candidate stays in range. `shrinking_preserves_a_dependent_bound` in
`src/runner/tests.rs` pins exactly that.

The cost is that a tape is opaque: a failure is reported by replaying it, not by pretty-printing
a stored shrink path. In exchange a persisted failure is a line of hex that reproduces exactly,
on any machine.

## Surface

**The numerical half** — why the crate is named for stability rather than for properties:

| module | role |
|---|---|
| `numeric` | condition numbers, forward and two-stage error bounds, seam hunting, ulp monotonicity scans |
| `tolerance` | ULP, relative and absolute comparison, and the condition number of a sum |
| `target` | a score for the search to maximise, so it hunts the worst case instead of sampling |
| `budget` | that worst score, committed — so losing accuracy fails even while the assertion holds |
| `relation` | the metamorphic relations, named, including `forward_stable` and `precision_agrees` |

**The property-testing foundation those sit on** — a search needs cases to search over and a way
to minimise what it finds:

| module | role |
|---|---|
| `source` | the choice tape; a value is a pure function of it |
| `generate` | generators and combinators, reading from a tape |
| `generate::float` | float generation built to find numerical bugs, not to be uniform |
| `shrink` | one generic minimiser over the tape, no per-type rules |
| `runner` | search, shrink, persist, report |
| `persist` | minimised failures as committable hex, with the case they render as |
| `stats` | case classification, with coverage you can *require* |
| `rng` | SplitMix64, the only randomness here |

Generators: `int`, `index`, `boolean`, `just`, `vec`, `distinct`, `permutation`, `one_of`,
`weighted`, `pair`, `triple`, `recursive`, plus `map` / `and_then` / `filter`. For floats:
`finite`, `any`, `magnitude` (log-uniform by exponent), `in_range`, `positive`, `notable`,
`non_finite`, `cancelling_pair`.

`distinct` and `permutation` are both "valid by construction rather than by rejection", which is
the house rule for a generator with a constraint: the shrinker's job is to make values smaller
and more alike, so it walks straight into whatever a `filter` rejects and spends its budget
there. `distinct` dedups (and so may come out short); `permutation` argsorts generated keys, which
is a genuine bijection every time and spread across the symmetric group — where "draw indices and
fix up the duplicates" biases towards the identity, the one permutation a broken `transpose`
survives.

Macros: `property!` to declare tests; `check!` / `check_eq!` / `check_ne!` / `check_ulps!` /
`check_relative!` to assert; `assume!` to skip a case; `classify!` to label one.

## Four things beyond the basics

**An accuracy budget.** A property asserts a contract — `check_ulps!(got, want, 8)` — and a
contract is written with headroom, so it admits today's error *and* a worse one. A refactor that
makes `erf` twice as inaccurate therefore passes every check: the property still holds. `target`
already hunts the worst case rather than sampling uniformly; `budget` writes that worst score to a
committed file and fails a later run that scores higher.

```rust
Config::default().budget("tests/budgets.txt")   // alongside target(observed_error)
```

```text
property regressed: the_erf_round_trip_is_forward_stable
  worst target() score rose from 3.0e-1 to 5.5e-1 (+84%)
  recorded in: tests/budgets.txt
```

The assertion stays the contract; the baseline notices the day the contract starts being used up.
It is only meaningful because the seed is fixed — the score is a property of the code, not of the
run — and the number lives in a file rather than a constant for the same reason a persisted
counterexample does: it is evidence produced by a run, and it should arrive in a diff as one line
rather than buried in a rewritten assertion. Score error, not accuracy, and prefer a ratio to an
admissible bound (`relation::forward_stable_margin`) over a raw error: `1.0` then means "exactly
at the limit", comparable across kernels.



**Metamorphic relations, named.** The same dozen shapes recur — involution, idempotence, round
trip, commutativity, associativity, distributivity, two-implementations-agree, monotonicity,
odd/even symmetry, invariant preservation, fixed point. Each is a function taking a
`Compare` (`Exact`, `Ulps(n)`, `Relative(t)`, `Absolute(t)`), so one relation serves a graph and
an `f64`:

```rust
involution("transpose", graph, |g| g.transpose(), &Exact)?;
involution("reciprocal", x, |v| 1.0 / v, &Ulps(2))?;
```

The reason to have them is not brevity. It is that the degenerate cases are handled once: a
hand-written commutativity check compares with `==`, which is wrong for a float and silently
wrong for a `NaN`, and that is a mistake that looks like a passing test.

**Coverage you can require.** `classify!` labels the case; `Config::require(label, percent)`
makes the run **fail** when the generator never produced it. Reporting the distribution is what
ScalaCheck's `classify` and jqwik's statistics do, and it is not enough, because nobody reads a
passing test's output. "At least 1% of cases must be subnormal" turns an assumption about the
generator into an assertion about it.

**Targeted search.** The property calls `target(score)` with how bad the case is — for a kernel,
the observed ULP error — and the runner spends half its budget perturbing the worst tape it has
found. That changes the question from "does this hold on 256 uniform draws" to "how badly can
this be made to fail". Hypothesis's `target()`; for numerics it is the most useful feature here.

Targeting never weakens a property: it changes which cases are tried, not which ones pass.

## Three outcomes that are not failures

Conflating these is how a suite reports more than it tested, so each is distinct and each is
surfaced:

- **Rejection** — a `filter` or `assume!` excluded the case. Tracked against a budget, because a
  generator rejecting most of what it makes is testing a fraction of its case count. Rejecting
  *everything* is an error, not a pass.
- **Overrun** — a tape did not feed the generator to completion. Only reachable while shrinking,
  where it means "discard this candidate", never "the property holds".
- **Budget reached** — shrinking stopped on its probe cap rather than at a fixpoint, so the
  reported case may not be minimal. Said out loud, since a reader otherwise assumes minimal.

## Determinism

The seed is fixed by default. A suite that passes today passes tomorrow, and a property that
only fails on one seed in ten is a property that fails — reporting that as a flake wastes the
finding. Vary `Config::seed` deliberately for a longer soak; do not vary it per run.

Set `EUCLID_STABILITY_VERBOSE=1` to let panic backtraces through during a search; they are
suppressed by default because shrinking re-runs a failing property hundreds of times.

## Not here

No `Arbitrary` derive — declare the generator, in the same spirit as `ConformanceSample` over a
blanket impl. No regex string generation, no process forking for aborts, no `no_std`. Each is
absent because nothing in the fleet needed it, and adding one later is cheaper than maintaining
it unused.

## Why a test harness lives in euclid

It does not meet euclid's membership rule, and does not claim to. Two projects using different
property runners do not produce two different numbers in production; they produce tests of
differing quality, which is a real cost but a smaller and different one.

It is here on the precedent `euclid-utils` sets — a `publish = false`, dev-only member that
never reaches a published artifact. The alternative was a separate repository for test tooling,
which is a lot of ceremony for eight hundred lines that only run under `cargo test`.

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
Copyright 2026 Raymond Mead.
