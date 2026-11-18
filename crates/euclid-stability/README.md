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
is immediately as shrinkable as everything else. Generators must still be deterministic and
re-derive dependent bounds after tape edits: a successfully generated case must remain legal.
`src/tests/tapes.rs` checks replay, bounded choices and dependent edits over generated inputs.

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
`src/runner/tests/search.rs` pins exactly that.

The cost is that a tape is opaque: a failure is reported by replaying it, not by pretty-printing
a stored shrink path. In exchange a persisted failure is a line of hex that reproduces exactly,
on any machine.

## Lessons from QuickCheck

[QuickCheck](https://github.com/BurntSushi/quickcheck) separates generation, shrinking and
execution. Its [Arbitrary trait](https://github.com/BurntSushi/quickcheck/blob/master/src/arbitrary.rs)
provides generation and an iterator of smaller values. Euclid instead declares generators with
explicit bounds and shrinks their choice tapes. Both designs need bounded shrink work and
reproducible generation; seeded generation alone does not preserve a case if its generator changes.

The [runner](https://github.com/BurntSushi/quickcheck/blob/master/src/tester.rs) separates passed
cases from discarded attempts and supports panics as failures. Euclid keeps those distinctions:
`cases` counts accepted executions, `max_rejects` bounds extra draws, and an exhausted rejection
budget fails the run. QuickCheck defaults `min_tests_passed` to zero; euclid requires the requested
case count. Tests exercise these guarantees directly instead of inferring them from passing laws.

The engine tests are organized by responsibility under
`src/{runner,relation,numeric,machine}/tests/`, with acceptance controls beside rejection cases.
`src/tests/tapes.rs` checks encoding, bounded draws, replay and dependent tape edits over generated
inputs. Consumer properties live with their owning contracts or numerical implementation.

The local `python3 -B bin/mutants.py` wrapper restores
regression and budget fixtures between mutation scenarios; a broken generator's saved tape must
not make a later mutant appear caught. Use fresh worker copies for each measured scenario when
the wrapper is unavailable. See [Contributing](../../CONTRIBUTING.md#the-seventh-check-run-deliberately-cargo-mutants).

## Running and inspecting a property

`check` is the libtest wrapper: it panics with a report on failure. `run` returns
`Result<RunReport, RunError>` so tools and engine tests can inspect the evidence directly.
`Outcome` remains `Result<(), Failure>`; existing property bodies and `?` still work.

```rust
use euclid_stability::{Config, assume, classify, generate, run, target};

let report = run("positive squares", &Config::default().cases(20),
    &generate::int(-10, 10), |x| {
        assume!(x > 0, "positive inputs only");
        classify!("positive");
        target(x as f64);
        assert!(x * x > 0);
        Ok(())
    }).unwrap();
assert_eq!(report.passed, 20);
assert_eq!(report.labels["positive"], 20);
assert_eq!(report.attempts, report.passed + report.rejected);
```

An assumption discards the case. Its labels and target scores are discarded too, even when
reported before `assume!`. The runner draws replacements until the accepted-case target is
met or the rejection budget is exhausted. A property that discards everything fails.

The report includes effective seed/configuration, attempts, passes, failures, rejections and
their reasons, regression replays, coverage and the best finite target score. Nested runs
preserve their parent's instrumentation. `RunErrorKind` distinguishes a counterexample from
generation, coverage, accuracy-budget, stale-regression and replay-consistency errors. A stable
property error includes the original and minimized tapes, final case, failure and shrink cost.

## Deterministic edges and small exhaustive domains

Random search remains the default, preserving existing tape layouts and accuracy budgets.
Choose another mode explicitly:

```rust
use euclid_stability::{Config, generate, run};

let flags = generate::array::<_, 3>(generate::boolean());
let report = run("boolean arrays", &Config::default().exhaustive(), &flags,
    |flags| { assert_eq!(flags.len(), 3); Ok(()) }).unwrap();
assert!(report.exhaustive);
assert_eq!(report.passed, 8);

let inputs = generate::int(100, 200).with_examples(vec![2, 3, 5, 7, 11]);
let mut seen = Vec::new();
run("examples", &Config::default().cases(5).edges_first(), &inputs,
    |x| { seen.push(x); Ok(()) }).unwrap();
assert_eq!(seen, [2, 3, 5, 7, 11]);
```

`edges_first()` tries complete edge tapes before random search. Integer ranges contribute their
endpoints and midpoint; `sampled_from` contributes every entry of a small list. `with_examples`
adds a sampled branch, so applying it deliberately changes a generator's tape layout. Edges
spend the normal case budget; choose enough cases to reach the edges you need.

`exhaustive()` visits every supported finite choice combination. Finite metadata composes
through ranges, fixed values, tuples, arrays, bounded lists, alternatives, maps and filters.
Filters and assumptions count rejected combinations separately; a many-to-one map can produce
duplicate values. An all-rejected domain still fails. Exhaustive mode returns
`UnsupportedExploration` when completeness cannot be established within the configured case
and tape budgets; it never silently falls back to random search.

Metadata construction is lazy and bounded: at most 4,096 tapes and 262,144 tape words per
collection; list enumeration supports maximum lengths up to 64. Large example lists use
representative edges. Custom `Gen::new`, dependent `and_then` and `sized` generators currently
have unknown exhaustive domains. These metadata limits do not constrain ordinary generation.

## Surface

**The numerical half** — why the crate is named for stability rather than for properties:

| module | role |
|---|---|
| `numeric` | condition numbers, forward and two-stage error bounds, seam hunting, ulp monotonicity scans |
| `tolerance` | ULP, relative and absolute comparison, and the condition number of a sum |
| `target` | a score for the search to maximise, so it hunts the worst case instead of sampling |
| `budget` | that worst score, committed — so losing accuracy fails even while the assertion holds |
| `relation` | the metamorphic relations, named, including `forward_stable` and `precision_agrees` |
| `machine` | sequential properties: a system against a model, over a generated operation sequence |

**The property-testing foundation those sit on** — a search needs cases to search over and a way
to minimise what it finds:

| module | role |
|---|---|
| `source` | the choice tape; a value is a pure function of it |
| `generate` | generators and combinators, reading from a tape |
| `generate::float` | float generation built to find numerical bugs, not to be uniform |
| `generate::float32` | native f32 bit patterns, signed zeros, subnormals and bounded ranges |
| `shrink` | one generic minimiser over the tape, no per-type rules |
| `runner` | search, shrink, persist, report |
| `persist` | minimised failures as committable hex, with the case they render as |
| `stats` | case classification, with coverage you can *require* |
| `rng` | SplitMix64, the only randomness here |

Generators: `int`, `index`, `boolean`, `just`, `vec`, `distinct`, `permutation`, `one_of`,
`weighted`, `pair`, `triple`, `recursive`, `sampled_from`, `array`, `option`, `result`,
`uint`, full-width `u64` / `u128` / `i128`, and `sized`, plus `map` / `and_then` / `filter`.
`sized(max, build)` records a shrinkable structural size independently of numeric ranges:
`sized(8, |n| vec(int(-100, 100), n, n))` generates a list of exactly the chosen length.
For f64 floats:
`finite`, `any`, `magnitude` (log-uniform by exponent), `in_range`, `positive`, `notable`,
`non_finite`, `cancelling_pair`, `binade_boundary`, `pair_with_edge_cases`.

The last two exist because of inputs the others structurally cannot produce, and both were added
after mutation testing found a defect a generated property had missed:

- **`binade_boundary(min_exp, max_exp)`** draws an exact power of two, both signs. `magnitude`
  deliberately fills all 52 mantissa bits — "round numbers hide alignment bugs" — which makes a
  binade boundary a `2^-52` event. That boundary is the only place where the step away from zero
  and the step toward it have different widths, so it is the only place a *direction* error is
  observable. `euclid-scalar`'s `ulp` had one; a property sweeping 1,000 decades did not see it.
- **`pair_with_edge_cases(min_exp, max_exp)`** draws an ordinary first element and a second from
  the cases a two-argument guard must handle — both NaNs, both infinities, both zeros, and the
  first element itself for the equality arm. A guard like `x.is_nan() || y.is_nan()` reads
  symmetrically and is not: weakened to `&&`, a NaN *first* argument can still produce a NaN by
  accident, while a NaN *second* argument returns a plausible finite neighbour. A property that
  fixes the second argument cannot see it.

`distinct` and `permutation` are both "valid by construction rather than by rejection", which is
the house rule for a generator with a constraint: the shrinker's job is to make values smaller
and more alike, so it walks straight into whatever a `filter` rejects and spends its budget
there. `distinct` dedups (and so may come out short); `permutation` argsorts generated keys, which
is a genuine bijection every time and spread across the symmetric group — where "draw indices and
fix up the duplicates" biases towards the identity, the one permutation a broken `transpose`
survives.

Macros: `property!` to declare tests; `check!` / `check_eq!` / `check_ne!` / `check_ulps!` /
`check_relative!` to assert; `assume!` to skip a case; `classify!` to label one.

## Five things beyond the basics

**Sequential properties.** Everything else here tests one value at a time, which cannot express a
bug that lives in an *interleaving* — a cache returning a stale entry only after an eviction, a
key tree colliding only after a split and a fold. `machine` takes a **model** (a slow
reimplementation whose correctness is obvious) and a **system** (the real thing), generates a
sequence of commands, and requires agreement at every step.

```rust,ignore
check_machine::<MyMachine>("a_name", &Config::default(), 12);
```

The reason this is cheap here and awkward elsewhere is the tape. Shrinking a command sequence
means deleting commands, and a deletion usually makes the rest **illegal** — popping a stack that
was never pushed. A framework holding a stored list has to detect and repair that, which is where
proptest-state-machine filters and Hypothesis re-checks preconditions. Here the commands after a
deletion are not replayed, they are *re-derived*: `Machine::command` takes the model, so the draw
reads the same tape entries against the state that now exists and produces whatever **that** model
offers. Every candidate is legal by construction. `an_arbitrary_tape_edit_still_produces_a_legal_sequence`
pins it by deleting and zeroing every entry of a tape in turn and executing the result.

The restriction that buys it: a command may depend on the model, never on what the system
returned. That keeps the sequence a pure function of the tape, and therefore shrinkable,
persistable and replayable — a sequential property inherits the whole engine rather than needing a
second runner.

## Four more things beyond the basics

**An accuracy budget.** A property asserts a contract — `check_ulps!(got, want, 8)` — and a
contract is written with headroom, so it admits today's error *and* a worse one. A refactor that
makes `erf` twice as inaccurate therefore passes every check: the property still holds. `target`
already hunts the worst case rather than sampling uniformly; `budget` writes that worst score to a
committed file and fails a later run that scores higher.

```rust,ignore
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

```rust,ignore
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

## Performance

The measurements below predate the structured runner and exploration metadata. Run the
benchmark again when assessing current throughput.

`cargo bench -p euclid-stability` prints a table. It is a plain `fn main` with `harness = false`
— no criterion, because a benchmark dependency is still a dependency and `deny.toml` would need
an exception for it.

| result | kind |
|---|---|
| shrink: **378 distinct probes, 194 answered from the memo — 33.9% avoided** | counted, exact |
| `Source::consumed()` per case removed: **9.3–9.5 ns, 4.2× the draw it sat beside** | interleaved A/B |
| per-case `Source` construction removed: **6.0–6.2 ns, 3.1×** | interleaved A/B |
| per-probe `Source` construction removed: **4.4–8.1 ns, 1.1×** | interleaved A/B |
| search floor ~59M cases/s; `Vec<f64>` generator ~5.4M cases/s | for scale only |

The per-case floor decides how many cases a cheap property can afford, which is where searching
harder is free. For an expensive property the engine disappears into the body, and the number
that matters is the probe count — because **shrinking a failure is routinely an order of
magnitude more work than finding it**, each probe re-running the generator *and* the property.

Four things got it there. The search no longer materialises the tape on every case (it is taken
only on a failure or a new best score, having previously been allocated, copied and discarded
every time). One `Source` is reused across all cases, and another across all probes in a
minimisation, rather than constructed per case and per probe. The panic-hook guard is a relaxed
load rather than a `SeqCst` read-modify-write per case. And the shrinker memoises its verdicts —
a third of all probes, because a fixpoint loop's final round exists to prove nothing more can be
done and proves it by re-trying candidates earlier rounds rejected. A memo hit also **does not
spend budget**, so the same `MAX_PROBES` cap explores strictly more distinct candidates:
shrinking got better as well as faster.

**Note which rows are counts and which are timings.** The probe count is exact and identical on
any machine. The A/B row runs both arms in the same process microseconds apart, and reproduces to
within 10% across runs. The throughput rows are for scale only: on the development machine they
varied by 2× between runs of *identical* code, which is more than any micro-optimisation here
does — the first version of this table quoted a 2× search speedup read off two separate runs, and
it was machine state rather than a result. If two arms cannot be interleaved, count something
instead.

**Threads are deliberately not part of this.** libtest already runs every `#[test]` on its own
thread and every property is a `#[test]`, so a suite is already using every core; parallelising
within a property would compete for the same cores rather than adding any. It would also cost
the fixed seed — a parallel search must either partition the seed space, invalidating every
recorded baseline, or accept a nondeterministic one — and that determinism is what makes a
budget baseline meaningful at all.

A note on reading the benchmark: it reports the **minimum** of seven passes after a warm-up, with
the spread alongside. A single timed pass is not a measurement — the first version of the harness
did exactly one and reported a 4× swing on unchanged code, which is enough noise to read a 34%
improvement out of nothing.

## Three outcomes that are not failures

Conflating these is how a suite reports more than it tested, so each is distinct and each is
surfaced:

- **Rejection** — a `filter` or `assume!` excluded the case. Tracked against a budget, because a
  generator rejecting most of what it makes is testing a fraction of its case count. Rejecting
  *everything* is an error, not a pass.
- **Overrun** — a tape or generation budget did not feed the generator to completion. It means
  "discard this candidate", never "the property holds".
- **Budget reached** — shrinking stopped on its probe cap rather than at a fixpoint, so the
  reported case may not be minimal. Said out loud, since a reader otherwise assumes minimal.

## Determinism

The seed is fixed by default. A suite that passes today passes tomorrow, and a property that
only fails on one seed in ten is a property that fails — reporting that as a flake wastes the
finding. Vary `Config::seed` deliberately for a longer soak; do not vary it per run.

Set `EUCLID_STABILITY_VERBOSE=1` to let panic backtraces through during a search; they are
suppressed by default because shrinking re-runs a failing property hundreds of times.

Before shrinking and again before saving, the runner checks that a failing tape reproduces
the same Debug rendering and failure identity at least twice (`Config::replay_checks`).
Assertions default to their source location; panics use the panic location. Shared helpers
can set `Failure::with_identity` to distinguish their laws. Shrinking keeps that identity,
so a smaller input triggering another assertion cannot replace the original bug.

Inconsistent replays produce `RunErrorKind::Unstable` without saving a counterexample. These
bounded checks catch common state leaks; they cannot prove purity. Generators, properties
and case formatting must remain deterministic, without time, external randomness or mutable
state affecting replay.

## Not here

No `Arbitrary` derive — declare the generator, in the same spirit as `ConformanceSample` over a
blanket impl. No regex string generation, no process forking for aborts, no `no_std`. Each is
absent because nothing in the fleet needed it, and adding one later is cheaper than maintaining
it unused.

## Where this goes next

Four open questions, in the order they constrain each other. Each one is here because a specific
piece of work ran into it, and the evidence is cited rather than summarised — a roadmap item
without a failure behind it is a preference.

### 1. `publish = false` makes every contract crate unpublishable

The decision that gates the others, and it is already load-bearing rather than hypothetical.
Every contract crate wires the laws behind a feature:

```toml
conformance = ["dep:euclid-stability"]
euclid-stability = { workspace = true, optional = true }
```

So:

```console
$ cargo publish --dry-run -p euclid-algebra
error: failed to prepare local package for uploading
Caused by:
  no matching package named `euclid-stability` found
  location searched: crates.io index
```

This crate is `publish = false` and the five contract crates therefore cannot go to crates.io at
all. Nothing is broken today — euclid is consumed by git tag, and [the README says
so](../../README.md#using-it) — but it is a one-way door.

**The feature is now split, and that was necessary without being sufficient.** `conformance`
keeps the fixed-sample laws and takes no dependency; `conformance-generated` pulls this crate.
`cargo tree --edges normal --features conformance` went from one `euclid-stability` edge to zero,
so a consumer taking the laws compiles and downloads no engine. Worth having on its own merits.

It does not make anything publishable, and the first pass at this section said it would. Measured
by removing one dependency table at a time:

| `euclid-stability` appears as | `cargo publish` |
|---|---|
| optional `[dependencies]`, feature **off** | **fails** |
| versioned `[dev-dependencies]` | **fails** |
| path-only `[dev-dependencies]` (no version) | stripped — passes |
| absent from both | passes |

Cargo resolves optional dependencies and versioned dev-dependencies from the registry at publish
time regardless of features. So the entry's *existence* is the blocker, not its activation, and
no amount of feature carving reaches it.

That leaves two real options, and the split has made the first one cheaper rather than replacing
it:

- **Publish this crate.** Now the obvious path. It costs a public API and a compatibility burden
  on a test engine whose posture is "shaped by what the fleet needed", and `0.x` churn here would
  propagate through the layer-1 chain in
  [CONTRIBUTING](../../CONTRIBUTING.md#what-a-release-costs-everyone-else). What the split buys is
  that the burden lands only on consumers who opt into `conformance-generated` — everyone else
  stops seeing this crate entirely, which is a much smaller promise than before.
- **Move the generated suites out of the contract crates.** Into a separate unpublished crate
  depending on both, or down into each consumer's own tests. That keeps this crate unpublished
  and costs the property RELEASES 0.5.0 advertised — that the generated strategy ships *with* the
  contract — plus a path-only dev-dependency everywhere.

Staying git-only remains defensible, but it should then be written in `RELEASES.md` as a decision
rather than left as a consequence.

### 2. References are the half this crate does not have

The engine finds inputs that violate a property. Accuracy checks need expected
values from an independent source.

Closing `BETA_EDGEWORTH_MIN` and checking the quantile solvers required independent Python
`Decimal` references. The earlier tables name their local generator. The newer
[extreme-shape generator](../euclid-scalar/tests/references/extreme_shapes.py) ships with the
scalar tests and uses MPFR and Decimal.

- **Commit new generators beside the owning crate's tests.** Name the generator from the table.
- **A generator self-validates before it emits**, against closed forms it shares no code with,
  and exits nonzero otherwise. This is not ceremony: `qref.py` caught a factor-of-two error in
  its own *expected* value that way — `I_x(2,3)` written with `B(2,3)⁻¹ = 6` instead of `12` —
  before a single reference was taken from it.
- **Provenance travels with the value**: what computed it, what it was validated against, at what
  working precision, and *at which exact double*. The last one is not pedantry — the beta's
  density is `≈3568`, so one ulp of `x` is worth `4e-13` of probability, and a reference computed
  at the real-valued argument would have charged the kernel `44×` its actual error.

### 3. Insensitivity, as a named relation

`relation` names seventeen metamorphic shapes — involution, idempotence, round trip, `agree`,
`precision_agrees`, `fixed_point`. It has no way to state the opposite: **this output must not
depend on that input.**

That is a real contract and euclid has one. `gamma_lr_inv` and `beta_reg_inv` are safeguarded
Newton: a seed, a bracket, and bisection whenever a Newton step would leave it. Their answers are
therefore independent of the seed *and* of the derivative, which was verified by disabling both —
seed removed, densities forced to `0`, so no Newton step can fire — and measuring against the
120-digit reference:

| | worst relative error vs truth |
|---|---|
| as shipped | `1.1e-13` gamma, `1.3e-14` beta |
| seed and derivative disabled | `7.4e-14` gamma, `1.3e-14` beta |

Unchanged, and marginally better. This is why 56 mutants in that file are unkillable, and the
right response is not to chase them: it is to **state the guarantee**.

`relation::insensitive_to(name, x, perturb, f, compare)` **now exists**, with accept and reject
controls beside the other sixteen relations. One honest limitation, which the first draft of this
section glossed: the quantile solver's independence is **internal** — the seed is a local, not an
argument — so no relation taking `f` from outside can perturb it. Expressing that would mean
making the seed injectable, which is production surface added for a test, and the measurement
above is better evidence anyway. The relation serves insensitivity that is part of the *input*:
periodicity, invariance under a common scale, a representation detail that carries no value.

A second honest note: euclid's own kernels currently have no *untested* insensitivity to point it
at. Degree-based trig periodicity, the obvious candidate, is already covered by
`real_carriers/reduction.rs`. So this is vocabulary for consumers and for future properties,
which is what most of `relation` is — and the reason to add it now is that the gap was only
visible once a 56-mutant cluster turned out to be a design guarantee nobody had written down.

### 4. Two gaps that were discipline, not capability

Worth recording because the instinct was to propose features and the measurement said otherwise.
The `BETA_EDGEWORTH_MIN` defect survived a sweep that stopped at `a = 1e100` while the format runs
to `1.8e308`. Both tools for catching that already exist:

- `generate::float::notable()` already draws `f64::MAX`, `f64::MIN`, both subnormals and `2^53`,
  and `magnitude(min_exp, max_exp)` takes explicit exponent bounds.
- `classify!` with `Config::require(label, percent)` already fails a run whose cases never reach a
  labelled region — which is precisely "this sweep never got above `1e102`".

Nothing was missing. Nothing required the label. So the change went to
[CONTRIBUTING](../../CONTRIBUTING.md#organizing-property-tests) rather than into API — **a
property guarding a threshold branch requires a coverage label on both sides of it**, and a
magnitude sweep ends at the format's limit or says in a comment where it stopped and why. Both
rules are written there now. A new generator would have been a worse answer than a used one.

### What to keep refusing

The list above still stands, with two additions earned this round: **no arbitrary-precision
arithmetic in Rust** — references come from outside and are pasted in with their provenance — and
**no second engine**. The temptation with the reference story is to grow this crate an oracle
module; the convention is the deliverable, not more code.

## Why a test harness lives in euclid

It does not meet euclid's membership rule, and does not claim to. Two projects using different
property runners do not produce two different numbers in production; they produce tests of
differing quality, which is a real cost but a smaller and different one.

It is here on the precedent `euclid-utils` sets — a `publish = false`, dev-only member that
never reaches a published artifact. The alternative was a separate repository for test tooling,
which is a lot of ceremony for eight hundred lines that only run under `cargo test`.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2026 Raymond Mead.
