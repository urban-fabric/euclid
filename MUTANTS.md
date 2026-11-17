# Surviving mutants

What `cargo mutants` still gets past the suite, why, and what closing each cluster needs.

Kept because the survivor list was large enough that re-deriving the diagnosis from a raw report
costs more than reading it, and because almost all of it turned out to be **a few structural
causes repeated**, not a long tail of unrelated holes. A list of line numbers hides that; this
does not.

Run `cargo mutants` to regenerate, see `.cargo/mutants.toml` for the configuration and the
exclusions, and CONTRIBUTING.md for the workflow.

## Where it stands

| Crate | Mutants | Caught | Survivors |
|---|---|---|---|
| `euclid-quantity` | 2,764 | 2,687 | **0** |
| `euclid-units` | 190 | — | **0** |
| `euclid-tensor` | 140 | — | **0** |
| `euclid-utils` | 152 | — | **0** |
| `euclid-random` | 33 | — | **0** |
| `euclid-matrix` | 19 | — | **0** |
| `euclid-scalar` | 13 | — | **0** |
| `euclid-stability` | 886 | 624 | **3** (equivalent — see below) |
| `euclid-algebra` | 459 | 346 | **13** |
| `euclid-numeric` | 6,605 | — | **~1,000** |

`euclid-quantity` was **1,966 survivors of 2,768** when first measured — the worst in the
workspace — and `euclid-stability` was 124.

## The causes, in order of how much they accounted for

These are worth internalising rather than skimming: each one predicts where to look next, and
between them they explain nearly every survivor found so far.

### 1. A shared default body is covered; its hand-written twins are not

Five independent instances. `LinearQuantity::add` is one implementation serving every carrier,
and `euclid-units` says so — "one body serves every carrier in every consuming crate … each one a
single point of failure". Testing it once is correct. Then sixty-one units each write `impl Add`,
`impl AddAssign`, `impl Neg` separately and nothing ties them back.

| Shared, covered | Twins, uncovered |
|---|---|
| `LinearQuantity::add`/`sub`/`neg` | 61 operator impls |
| `fmt_quantity` | 64 `Display` impls |
| `scalar_mul` / `scalar_div` | 118 `Mul<f64>`/`Div<f64>` impls |
| `DiscreteQuantity::saturating_sub` | the inherent twins in `data/` |
| `ring_laws` (euclid-algebra) | the compound-assignment operators |

Closed by `operator_bridge_laws`, `scaling_laws` and the per-unit suites — all of which assert
the twin *agrees with the original* rather than testing each in isolation.

### 2. A cross-type surface no single-type law can reach

`quantity_laws` takes a `Q` and stays inside it. `Length::per_time` consumes a `Time` and returns
a `Speed`; `PartialEq<u64> for Bytes` compares across types; `Interval`/`Point` are generic
wrappers rather than units. Three clusters, closed by `dimensional_tests`, the `data/` interop
tests and `ops/arithmetic_tests`.

### 3. A parallel or fallback implementation nothing instantiates

- **f32 beside f64.** The suites instantiate at `f64`; the f32 tree rides along.
  `precision_agrees` exists for exactly this and was called **nowhere**.
- **Asymptotic branches behind a threshold.** `BETA_EDGEWORTH_MIN = 1e7`,
  `GAMMA_EDGEWORTH_MIN = 3e7`, `STIRLING_RATIO_MIN = 300`, `PSI_ASYMPTOTIC_MIN = 20`. No test
  input crosses any of them.
- **Carriers beside the reference one.** `natural_impl`, `integer_impl`, `boolean_impl`.

### 4. A generated suite keyed on a shape some types do not have

Two instances, both found by the measurement rather than by inspection — which is the argument
for keeping the measurement. The `inherent_tests` generator keys on `as_<unit>() -> f64`, so the
integer-backed discrete time units fell through every net. And `dimensional_tests` reads the
operation off the `per_`/`times_` naming convention, so `MolarVolume::concentration` — a
reciprocal, named neither — was structurally outside it.

### 5. An affine conversion, where a round trip cancels the error

One instance: the temperature scales. `from_celsius(x).celsius() == x` proves nothing about the
offset, because the offset cancels. Closed by `temperature_laws`, anchored on water's phase
changes and absolute zero.

## Not one of them was a bug

Every law added across this work passed on first run. That is the most useful single fact here:
the instrument is reporting **where tests are missing**, not where code is broken. Treat a
survivor as a missing assertion until proven otherwise.

## Avoiding tautology, which took three tries to get right

A test generated from the code asserts that the code is the code. `notable_reaches_every_named_value`
had already fallen into this — it iterated the very constant it was meant to check — so three
clusters needed their expectation sourced from outside the implementation:

| Cluster | Independent source |
|---|---|
| SI exponent table | physics definitions (pressure is force per area) |
| dimensional methods | the `per_`/`times_` naming convention |
| temperature scales | physical fixed points (−40 °C = −40 °F) |
| time conversions | the SI prefixes |
| f32 kernels | the f64 result, correctly rounded |

## `euclid-numeric` — the remaining half

| Cluster | Count | Needs |
|---|---|---|
| `kernels/f32/quantile.rs` | 118 | its own cross-precision pass — `gamma_lr_inv`, `gamma_start`, `beta_reg_inv` are not on `Real` |
| `impls/real_f32.rs` | 117 | partly done; `hypot3`, `ln_sub_exp`, `sin_cos_degrees` need value tests, not budgets |
| `kernels/f32/gamma.rs` | 91 | — |
| `kernels/f32/polygamma.rs` | 77 | — |
| `kernels/erf/f32.rs` | 74 | — |
| `impls/natural_impl.rs` | 69 | instantiate the suites at the natural carrier |
| `kernels/quantile.rs` | 56 | — |
| `kernels/beta.rs` | 47 | accuracy cases above `BETA_EDGEWORTH_MIN` |
| `kernels/transcendental.rs` | 46 | — |
| `impls/boolean_impl/unsigned_compare.rs` | 46 | — |
| `kernels/bessel/modified_first.rs` | 41 | — |
| `impls/integer_impl.rs` | 34 | — |
| `kernels/gamma.rs` | 31 | cases above `GAMMA_EDGEWORTH_MIN` / `STIRLING_RATIO_MIN` |

`tests/identities/cross_precision.rs` pins measured f32-vs-f64 ulp budgets for thirty-five
kernels and the four mathematical constants. Three readings from that measurement are recorded
there and are worth not re-deriving:

- Many kernels are **bit-exact** because they compute in f64 and round once. A budget of zero
  across four hundred arguments is not luck, and pinning it would catch someone converting one
  into a genuine f32 path.
- The separately-implemented kernels sit at 35–69 ulp, and they are exactly the files the
  survivors concentrate in.
- `ln_gamma` at 1,514 ulp, `digamma` at 205 and `atanh` at 17 are **not defects**. Each has a
  zero inside its range, and ulp spacing collapses towards zero while the derivative does not —
  the same effect `euclid_stability::numeric::seam_ratio` documents and divides out.

Adding cases above the four thresholds is worth doing on its own merits, independent of mutation
score: those are the branches a caller reaches with real parameters, and nothing in the suite has
ever evaluated them.

## `euclid-algebra` — 13

All in `conformance.rs`, all the same vacuity pattern: a law suite replaceable with an empty body,
because every test runs it against a *correct* carrier. `star_laws`, `canonical_order_laws`,
`ordered_semiring_laws`, `left_semimodule_laws`, the two `witness_*`, and the `generated::*`
family with its `settings`/`cases`.

The template is `src/tests/vacuity.rs` — a carrier deliberately wrong in exactly one law, claiming
the marker it violates, with the suite required to panic. Cheap on the semiring trunk, where the
laws are marker traits; more work above it, where a tier has to be carried before it can be broken
in one corner. `Faulty<0..3>` shows the pattern for that.

## `euclid-stability` — 3, and they stay

Equivalent mutants that cannot be excluded by regex without hiding a real finding in the same
function, since `exclude_re` matches a mutation's description and not its line. The reasoning lives
on `seam_ratio` in the source, where someone reading a report will look: both `||`s in its
finiteness guard are early-outs the closing `ratio.is_finite()` makes redundant, and
`x + window == x` mutated to `x - window == x` duplicates the clause on the next line.

**Do not "fix" these.** The other mutations of the same guard are real and are pinned — at
`x = -f64::INFINITY` three of them make the function report a finite ratio of zero for an argument
where nothing was measured.

## Two notes on using the tool

**`exclude_re` matches the mutation's *description*, not its source line.** That description reads
`<impl Trait for Type>::method`, not `Type::method` — patterns written the latter way match
nothing and fail silently. And because there is no line anchor, a pattern naming only an operator
and a function will hide every mutation of that shape, including future real ones. Name the
operator *and* check what else shares the name.

**A `mutants.toml` at the repository root is silently ignored.** Only `.cargo/mutants.toml` is
read. The run looks completely normal and applies none of the exclusions.
