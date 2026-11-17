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
| `euclid-numeric` | 6,605 | — | measured per file — see below |

`euclid-quantity` was **1,966 survivors of 2,768** when first measured — the worst in the
workspace — and `euclid-stability` was 124.

`euclid-numeric` is the one crate too large to re-measure in a sitting: 6,605 mutants at about
three seconds each is over five hours, so it is worked and verified **file by file** and the
table below carries the per-file numbers rather than a crate total. A full run remains the only
way to get one, and nothing here should be read as implying the untouched files have moved.

| File | Mutants | Was | Now |
|---|---|---|---|
| `impls/natural_impl.rs` | 159 | 69 | **0** |
| `impls/integer_impl.rs`, `impls/boolean_impl/**` | 258 | 103 | **0** |
| `kernels/host/factorial.rs` | 119 | — | **0** (119 caught) |
| `kernels/f32/**` | 1,239 | ~300 | **0** |
| `impls/real_f32.rs`, `kernels/erf/f32.rs` | 470 | 191 | **15**, every one characterised below |

So **575 survivors closed**, two of them defects rather than gaps.

Two caveats on reading that. The `0`s include three timeouts, which this file has always counted
as caught: a mutated loop that spins makes the suite hang rather than pass. And
`kernels/host/factorial.rs` has no "was" figure because it did not exist in its current form —
the exactness fix rewrote it, so its 119 mutants are a different population from the ones the
original measurement saw.

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

### 6. A refinement step computed with a coarser instrument than the thing it refines

Found in `euclid-numeric`, and it accounts for the largest single cluster in the crate — the
**thirty-four** mutants in the f32 `erf_inv`, which was every seed coefficient and every operator
in its correction step.

The shape, as it first appeared: `erf_inv` is a Giles rational approximation followed by **one
Newton step**, so the polynomial only chooses where the iteration *starts*. Newton is
quadratically convergent, so perturbing a seed coefficient ought to move the starting point and
leave the answer alone — which is a sufficient explanation for the survivors, and was the
recorded one for a while.

It was the wrong explanation, and measuring the function against the *truth* rather than against
itself is what showed it. At f32:

| | max relative error vs the true inverse |
|---|---|
| the Giles seed alone | **`2.4e-7`** — about two f32 ulps, as good as the format allows |
| seed + one Newton step, as shipped | **`1.9e-6`** — eight times worse |

**The Newton step was making it worse.** The correction is `err / pdf` with
`err = erf(result) - x`, and `erf` at f32 carries its own error of several ulps. Dividing that by
`2/sqrt(pi) * e^{-r^2}`, which collapses as `|r|` grows, amplifies it — so the step injected more
noise than the seed had error. Giles fitted those coefficients *for* f32; there was nothing left
to refine, and the refinement was computed with a coarser instrument than its target. In the far
tail it was merely useless rather than harmful: past `|p| ~ 0.997` the `e^{-r^2}` factor
underflows and the correction is identically zero.

So the thirty-four survivors were not reporting a seed hidden behind a correct correction. They
were reporting that the correction destroyed the seed's precision and then contributed nothing
reliable of its own — which left no accuracy bound able to separate any version of the function
from any other. Deleting it improved the kernel eightfold, removed eight mutants along with the
code, and made the remaining twenty-six observable, now pinned at `3e-7` by
`the_f32_inverse_is_as_accurate_as_the_format_allows`.

The f64 peer keeps its Newton step, and should: its coefficient table *is* coarser than f64, so
there the refinement is finer than its target. The asymmetry is the point — **the same algorithm
is right at one precision and wrong at the other** — which is causes 3 and 7 arriving in a third
form.

**Two near-misses worth recording,** because either would have ended the investigation with a
wrong answer:

- A first measurement appeared to show the round trip separating the mutants cleanly: unmutated
  `7.5e-7` against mutants at `4.2e-5` to `1.3e-3`. That harness reimplemented the seed and
  **left the Newton step out**, so it measured the seed rather than the function. A
  reconstruction has to be checked against the shipped code — the corrected harness was verified
  bit-for-bit over 8,000 arguments before anything was concluded from it.
- The round-trip identity `erf(erf_inv(p)) = p` looks like the right instrument and is not
  sensitive enough. It compares the function with *its own inverse*, so it cannot see an error
  the pair share; against the truth the gap is eight times larger than the round trip suggests.

`gamma_lr_inv` and `beta_reg_inv` are Newton/bisection solvers with their own seeds
(`gamma_seed`, `gamma_start`) and have not been examined this way. Given what `erf_inv` turned
out to be, they are worth measuring against the truth rather than assumed benign.


### 7. Dead precision — arithmetic the format cannot represent

Also `euclid-numeric`, and the counterpart to cause 3's "parallel implementation": the f32 body
is the f64 body with the type changed, and some of it no longer reaches the output.

`sinc`'s Taylor branch carried `1 − x²/6 + x⁴/120 − x⁶/5040` at both precisions. Below its `1e-2`
crossover, `x²` is at most `1e-4`, so the `x⁴/120` term is at most `8.3e-10` against a result
near one — seventy times finer than f32's `6e-8` resolution there. Carrying the last two terms
changed **none of 3,000 sampled results**. They were eight mutants that no test could ever kill,
because they were arithmetic with no observable consequence.

Closed by computing what f32 can actually represent: `1 − x²/6`, which is bit-identical across
the branch. This is exactly the failure `euclid_stability::relation::precision_agrees` is
documented to look for — "a polynomial fitted for f64 can be the wrong polynomial at f32" — and
the fix is a deletion rather than a test.

Worth checking for wherever an f32 kernel was derived from an f64 one: a term below the format's
resolution is both a performance cost and a permanent survivor.

## Almost none of them was a bug — and then two were

For the first three crates, every law added passed on first run. The instrument was reporting
**where tests were missing**, not where code was broken, and the standing advice was to treat a
survivor as a missing assertion until proven otherwise.

That advice still holds as a prior. But `euclid-numeric` produced two exceptions, and both are
worth knowing in detail, because neither looked like a bug from the outside and neither was found
by a failing test.

Both were found the same way: **not by reading the survivor, but by doing the work to decide
whether the survivor was equivalent.** That work means computing the right answer from an
independent source, and in both cases the independent source disagreed with the code.

### The off-by-one: `binomial` outside its documented exact range

**`binomial` was off by one, inside the range where its docs promised exactness.** `C(56, 29)` is
`7_384_942_649_010_080` — comfortably below `2^53` — and the kernel returned
`7_384_942_649_010_079`. Twelve of the 1,840 coefficients with `n <= 60` whose true value `f64`
can hold exactly came back wrong, every one of them low by one.

The docstring reasoned: each partial product `C(n-k+i, i)` is an integer, so the division is
exact, so the recurrence is exact while the result is under `2^53`. The first two clauses are
true. The conclusion does not follow, because the bound is set by the *intermediate* and not by
the answer: `acc·(n-k+i)` equals `i·C(n-k+i, i)`, a factor of `i` larger than the partial it
produces, so an `f64` accumulator starts rounding a factor of roughly `k` below `2^53`. Fixed by
carrying the recurrence in `u128` and rounding once at the end.

Three things about how it was found are the transferable part:

- **No test failed, and no survivor pointed at it.** The surviving mutant was
  `replace - with + in binomial`, a mutation of the `k.min(n - k)` reduction. Investigating
  *whether that mutant was equivalent* is what required computing exact binomials, and the
  exact values disagreed with the kernel. The mutant was a signpost, not a diagnosis.
- **A reference sharing the implementation's algorithm would have agreed with the bug.** The
  test now uses Pascal's rule — addition only — against a kernel that multiplies and divides.
  This is the tautology trap from the section below, in its most expensive form: not a test that
  proves nothing, but a test that would have actively confirmed the defect.
- **The hand-written twin had drifted further.** `Natural for u128` carried its own copy of
  `binomial` with the same flaw, and its own `ln_binomial` differenced straight off `ln_gamma`
  where the shared body used the exact small sum — so `u64` and `u128` returned different
  answers to the same combinatorial question. All four combinatorial methods now delegate to one
  body.

### The counterproductive correction: the f32 `erf_inv` Newton step

The second is cause 6 in full, and only the conclusion belongs here: the f32 inverse error
function applied a Newton correction that made it **eight times less accurate** — `1.9e-6`
against the bare seed's `2.4e-7` — because the correction divided `erf`'s own f32 error by a
rapidly-vanishing derivative. Thirty-four mutants survived in that one function, and the reason
was not that they were harmless but that the correction had already destroyed the precision they
would have perturbed.

The same two lessons recur:

- **The survivor count was the signal, not the diagnosis.** Thirty-four in one function is not a
  gap in a test suite; it is a statement that nothing about the function's internals reaches its
  output. That is worth investigating as a *design* question.
- **The identity was not enough.** `erf(erf_inv(p)) = p` compares the function with its own
  inverse, so it cannot see an error the two share. Only comparing against the truth — the f64
  kernel, four orders of magnitude finer than f32 can represent — showed the step was harmful.

### What to carry forward

Still treat a survivor as a missing assertion first; that is right the great majority of the
time. But two refinements, both paid for:

1. **When deciding whether a mutant is equivalent, compute the right answer from an independent
   source.** That step is what found both bugs. Reasoning from the source found neither — in the
   `binomial` case the source's reasoning was itself the error.
2. **A large cluster in one small function is a design signal.** Twelve wrong coefficients, or
   thirty-four survivors in forty lines, says the function's internals do not reach its output.
   Ask why before writing a test.

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

## `euclid-numeric` — the carrier tiers and the f32 tree

### What has been closed

| Cluster | Was | Now | What closed it |
|---|---|---|---|
| `impls/natural_impl.rs` | 69 | **0** | `tests/discrete_carriers/natural.rs` — the whole of `Natural` at all three widths |
| `impls/integer_impl.rs` | 34 | **0** | `tests/discrete_carriers/integer.rs`, same shape with both signs |
| `impls/boolean_impl/**` | 62 | **0** | `mask_laws` extended from three widths to all six |
| `kernels/f32/quantile.rs` | 118 | **0** | `cross_precision/parameterised.rs` — swept per shape, not per argument |
| `kernels/f32/gamma.rs` | 91 | **0** | the same pass, at the shapes that select each branch |
| `kernels/f32/polygamma.rs` | 77 | **0** | the same pass, at each order |
| `kernels/f32/normal.rs` | 14 | **0** | the same pass |
| `impls/real_f32.rs` | 117 | small | `tests/real_carriers/` — values and round trips, not budgets |
| `kernels/erf/f32.rs` | 74 | small | `real_carriers/special_functions.rs` |

### The fifteen that stay, and why

`impls/real_f32.rs` and `kernels/erf/f32.rs` are down to two families, both of which
`euclid-stability` already keeps examples of. Each was checked individually rather than waved at,
and the checking is the reason one of them turned out to be real.

**Eight crossover comparisons, where both branches are right at the boundary.** A kernel that
switches algorithms at a threshold has, by construction, two expressions that agree there — that
is what makes the threshold safe to cross. Widening `<` to `<=` moves the switch by one
representable value, and at that value both branches return the same thing.

| Site | The two branches, at the boundary |
|---|---|
| `ln_1p_exp` at `x > 0` | `x + ln_1p(e^-x)` and `ln_1p(e^x)` both give `ln 2` at `x = 0` |
| `ln_1m_exp` (three) | measured `-0.693_147_2` against `-0.693_147_3` either side of `-ln 2` |
| `erf`, `erfc`, `erfcx` (three sites, six mutants) | the series and continued-fraction forms meet at `ERFCX_CF_MIN` |
| `erf_inv` at `w < 5` | the two Giles coefficient tables meet at `w = 5` |

**Five NaN and infinity guards, where the arithmetic underneath reaches the same answer.**
Exactly the shape documented for `within_relative` at the top of this file: weaken
`is_nan() || is_nan()` to `&&` and a lone NaN falls through, but every path below produces NaN
from the NaN anyway.

Not a free pass, and taking the family apart case by case is what mattered — **two of the six
were real**:

- `ln_add_exp`'s opening guard **is** killable. A NaN falling past it reaches
  `self == inf || other == inf`, which fires on the infinity and returns `+inf` for an input
  containing a NaN. Now pinned by `ln_add_exp(NaN, inf)`.
- `ln_sub_exp`'s `self == inf && other == inf` widened to `||` **is** killable, because
  `ln_sub_exp(inf, b)` for finite `b` is `inf` and the mutant returns NaN. Now pinned.
- The remaining four — `ln_add_exp`'s second guard and both of `ln_sub_exp`'s `||`s — are not.
  Every fallthrough path ends in arithmetic on a non-finite value and returns the same thing.

**One argument reduction invariant under its own mutation.** `sin_cos_degrees` reduces with
`self % 360.0`, and replacing that with `self + 360.0` adds exactly four to the quadrant index
`q`, which `q.rem_euclid(4)` then discards; the residual `r - 90*q` is unchanged. It becomes
observable only where `self + 360.0` rounds, which is past `10^9` degrees and past any argument
the function is meant for.

**One further mutation is excluded rather than counted here** — `sinc`'s crossover comparison,
where both the Taylor series and `sin(x)/x` sit within 1 ulp of the correctly-rounded result on
either side. Measured across 3,000 arguments at each precision.


### Still untouched

| Cluster | Count | Needs |
|---|---|---|
| `kernels/quantile.rs` | 56 | the f64 seeds, same cause-6 problem as their f32 peers |
| `kernels/beta.rs` | 47 | accuracy cases above `BETA_EDGEWORTH_MIN` |
| `kernels/transcendental.rs` | 46 | — |
| `kernels/bessel/modified_first.rs` | 41 | — |
| `kernels/gamma.rs` | 31 | cases above `GAMMA_EDGEWORTH_MIN` / `STIRLING_RATIO_MIN` |

Adding cases above the four thresholds is worth doing on its own merits, independent of mutation
score: those are the branches a caller reaches with real parameters, and nothing in the suite has
ever evaluated them.

### The cross-precision pass, now in two halves

`tests/identities/cross_precision/` is split because the two halves make opposite claims, and a
kernel moving between them is a real change of character that should show up in a diff:

- **`delegating.rs`** — computes in f64 and rounds once, so **bit-exact**, pinned at zero ulp.
  Also the constants, and `beta_reg`, which turned out to be bit-exact at every shape measured.
- **`parameterised.rs`** — separately implemented, so a **measured budget**. The multi-argument
  kernels live here, swept at several shapes each, because the shape selects the branch: a
  series below the crossover, a continued fraction above it, an asymptotic form past the
  threshold. One shape exercises one of those and leaves the rest unmeasured, which is why
  `kernels/f32/quantile.rs` had the largest cluster in the crate and nothing reached it.

Three readings from the measurement are worth not re-deriving:

- Many kernels are **bit-exact** because they compute in f64 and round once. A budget of zero
  across four hundred arguments is not luck, and pinning it would catch someone converting one
  into a genuine f32 path.
- The separately-implemented kernels sit at 35–69 ulp, and they were exactly the files the
  survivors concentrated in.
- `ln_gamma` at 1,514 ulp, `digamma` at 205 and `atanh` at 17 are **not defects**. Each has a
  zero inside its range, and ulp spacing collapses towards zero while the derivative does not —
  the same effect `euclid_stability::numeric::seam_ratio` documents and divides out. The same
  reading explains `gamma` at 387 ulp on the negative axis, which crosses a pole between every
  pair of consecutive negative integers, and `ln_rising_factorial(a, 0.5)` at 301, which passes
  through zero near `a = 1.23`.

### A constant checked is a constant chosen

`the_constants_agree_across_precisions` covered four of the trait's twenty-four, and `sqrt_tau`
surviving a mutation to `Default::default()` is what that cost. Extending it to everything
immediately found something the selective list had hidden: **`degrees_per_radian` is derived, not
lifted.** Its body is `180 / pi`, so each carrier divides using its own `pi` and the f32 value
(`57.295_776`) sits thirty-two ulp from the narrowed f64 value (`57.295_780`). Not a defect —
putting a rounded `pi` in a *denominator* carries the error into the quotient — but not a
bit-exact relationship either. Its sibling `radians_per_degree` agrees bit-for-bit, which is
precisely the coincidence that makes a partial list look fine.

The two derived constants are now checked against the true value at each carrier's own
resolution, which is the only claim either can support.

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

## Three notes on choosing the instrument

The `euclid-numeric` work turned on picking the right comparison, more than on writing more
assertions. Three cases where the obvious instrument was the wrong one:

**A cross-precision budget cannot pin a separately-fitted coefficient.** `erf_inv` at f32 and at
f64 use *different* approximations, so they legitimately disagree by up to 15 ulp and the budget
has to be 15. Its early coefficients contribute about `7e-7` relative — inside that bound. The
budget is the right instrument for a kernel that computes in f64 and rounds once, and the wrong
one for a kernel with its own fit. Check which kind you have before trusting the number.

**A tolerance loose enough for the real code can be loose enough for the mutant.** At `n = 21`
the f32 `factorial_u32` takes its `exp(ln_gamma)` branch with `4.3e-6` relative error, while the
loop branch the mutant selects is *exact*. Any accuracy bound admitting the real behaviour
admits the mutant too, so the crossover had to be pinned by value. When a mutant is **more**
accurate than the code, only an exact assertion separates them.

**An identity between two functions is circular if they share a body.** `erfcx(x) = e^{x²}
erfc(x)` looks like a free cross-check. Below `ERFCX_CF_MIN` it *is* how `erfcx` is computed, and
above it both route through the same continued fraction — so it is the implementation restated,
at every argument. The independent reference there is a series, and the series is only
trustworthy where it does not cancel, which turned out to be the same `x = 2` boundary.

## Two notes on using the tool

**`exclude_re` matches the mutation's *description*, not its source line.** That description reads
`<impl Trait for Type>::method`, not `Type::method` — patterns written the latter way match
nothing and fail silently. And because there is no line anchor, a pattern naming only an operator
and a function will hide every mutation of that shape, including future real ones. Name the
operator *and* check what else shares the name.

**A `mutants.toml` at the repository root is silently ignored.** Only `.cargo/mutants.toml` is
read. The run looks completely normal and applies none of the exclusions.
