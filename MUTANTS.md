# Surviving mutants

What `cargo mutants` still gets past the suite, why, and what closing each cluster needs.

Kept because the survivor list was large enough that re-deriving the diagnosis from a raw report
costs more than reading it, and because almost all of it turned out to be **a few structural
causes repeated**, not a long tail of unrelated holes. A list of line numbers hides that; this
does not.

See `.cargo/mutants.toml` for the configuration and exclusions, and CONTRIBUTING.md for the
workflow. Commands under `bin/` describe the local tools used to produce earlier measurements.
The 0.6.0 tree excludes that directory.

## 0.6.0 release audit

The workspace enumerated 11,485 candidates with `--workspace`. The changed production lines
in the scalar kernels, rational arithmetic and quantity midpoint operations exercised **269**
mutants against the all-features workspace suite. **255** were caught by tests, **12** failed to
compile and **2** timed out. The timeouts changed termination in rational comparison and digamma.
Every viable mutation in these scopes failed a test or timed out.

| Scope | Tested | Caught | Unviable | Timed out |
|---|---:|---:|---:|---:|
| Distribution kernels, rational arithmetic and quantity midpoints | 266 | 252 | 12 | 2 |
| `f32` inverse complementary error function | 3 | 3 | 0 | 0 |

The single-precision gamma and beta quantiles share one Newton and bisection loop. A replay of
20,000 seeded input tuples against the starting checkout compared 80,000 outputs across both
distributions and precisions. All matched bit for bit for shapes between `0.125` and `512`,
including sampled tail probabilities. The extreme-input properties cover the intentional changes
at the format limits and require coverage on both sides of the relevant boundaries.

The rational comparator keeps a cross-product fast path for components that fit in `i32`.
In an optimized microbenchmark of 6.55 million comparisons of small `Rational64` values, the
median rose from roughly 4.0 ms to 7.2 ms. The wider comparison is covered by an `i128`
cross-product oracle for `Rational64` and adjacent-ratio checks for `Rational128`.

Rust 1.95 and the available stable toolchain passed all 1,674 workspace tests. The optimized
workspace build also passed. Two isolated downstream copies exercised the 0.6 migration.
Bayes passed 829 tests with its original accuracy budgets after the `f32` inverse-erfc fix.
Replaying the 239 distinct inputs from the old and new Bayes searches found a change in the
same-input normal quantile. The regression now brackets the root between rounding midpoints,
including the smallest `f32` probabilities. Hausdorff passed 1,569 tests after applying the
documented discard semantics to its grid tests and fixing its S2 test to check the position of
the least significant set bit. The migration checks used local copies of the downstream sources.

### Follow-up on the reproduced release risks

- With `m = i64::MAX`, `1/m + 1/(m/7)` now returns `8/m`. The checked rational API
  distinguishes an exact reduced result from overflow and a zero divisor. A debug and release
  fixture exercises signed minima and widened `Rational128` addition. An independent Python
  `Fraction` replay matched 22,800 full-width and cancellation-heavy operand pairs.
- At `x = 1`, `gamma_ur(1e-320, x)` now rounds to `2.194e-321`. `ln_beta(1e306, 1e306)`
  returns `−1.3862943611198906e306`. The shifted beta approximation matches the finite sum
  for integer first shape across the tested extreme cases. The new fixtures also cover
  upper gamma quantiles and the smallest positive input to the double-precision inverse erfc.
  Regenerate the independent references with
  `python3 -B crates/euclid-scalar/tests/references/extreme_shapes.py`.
- The matrix residual helper now rejects `[0]` versus `[1e200]` and handles subnormal
  absolute errors. Tests instantiate the public helper on a rank-zero eager carrier and
  check accumulated norm factors independently. The decomposition and solve laws still
  need a backend supplying those primitives.

On the follow-up, 1,698 workspace tests passed on stable and Rust 1.95 and in the optimized
profile. The isolated Bayes and Hausdorff builds passed 829 and 1,569 tests respectively.

A 524-mutant run over the changed algebra, distribution and matrix routines found 488 caught,
28 unviable, two timeouts and six survivors. The two surviving wide-integer bit operations are
equivalent: in the product, the left limb has 64 trailing zeros and the right limb contributes
only those 64 bits. In division, the shifted remainder has a zero low bit and the other
operand is one bit. OR and XOR therefore yield the same integer at both sites. The remaining
findings led to added tests or changes to the corresponding paths. The final tail and residual
follow-up caught 95 of 96 mutants. The remaining case exposed a redundant domain guard and a
missing invalid-probability test. The final single-precision gamma inverse check caught all
12 mutants. These scopes overlap.

The large-shape beta CDF now standardizes the compensated residual `x(a+b) − a`.
With `a = f64::MAX`, `b = a.next_down()` and `x = 0.5`, the CDF saturates to 0
in the first shape order and to 1 in the second. A one-sided variance bound puts the
far tail below `1e-270`. The regressions cover both orders, an exact asymmetric mean,
and finite shape sums for which adjacent binary64 inputs span the central transition.
When the shifted gamma limit would itself use Edgeworth, the CDF standardizes
directly in beta coordinates to preserve the residual at extreme shape ratios.
An exact rational replay of 126,336 near-mean inputs agreed on the residual's sign
and equality to zero in every case.

### Shared scalar steps

`euclid-scalar` keeps compensated arithmetic, relative-term checks, continued fractions,
Euclidean arithmetic and bracketed Newton steps in its private `math` module. Exact dyadic
cases exercise equality and adjacent floating-point values. The series test records which
terms the kernel requests, and the LCM test verifies that a zero operand skips GCD entirely.
These checks and the shared implementations retired nine `exclude_re` patterns.

With exclusions disabled, the changed scalar lines yielded 254 caught mutants, 14 compile
failures and one survivor. Widening the Newton test from `density > 0` to `density >= 0`
admits signed zeros, but dividing by zero produces an infinite or NaN candidate that the
finite bracket rejects. The previous GCD timeout now fails to compile because the Euclidean
step requires remainder arithmetic and has no division bound. A separate run on compensated
dot products, interval gaps and compensated series caught 23 mutants and rejected two at
compile time. Both runs finished with zero timeouts.

A replay of 65,536 beta and gamma CDF and quantile results matched the prior candidate bit for
bit. `Ei(710)` now returns about `3.1509156882062014e305`. MPFR fixtures cover the largest
finite-result input and the adjacent input that overflows. The Rust 1.95 debug and release
workspace suites each passed 1,750 tests and 38 doctests.

### Formula decomposition and numerical checks

The shared Horner evaluator uses `Semiring`, and the two arithmetic forms of the logarithmic
complement use `Real`. The host kernels choose the evaluation regime. The sinc tests compare
Horner against exact rational factorial terms and check the cutoff at adjacent floats. The
beta continued fraction tests cover both orientations against an integrated polynomial and
check its prefactor separately. The direct sinc cutoff checks replace two exclusion patterns.

Near `ln(2)`, `ln(exp(x) - 1)` now evaluates `ln1p(2 * expm1(x - ln(2)))` using a split
`ln(2)`. The high-limb subtraction is exact in `[0.5, 1]`. The fixtures use a 500-digit
Decimal evaluation of the original expression, including both floating-point neighbors of
the rounded `ln(2)`. Regenerate them with `crates/euclid-scalar/tests/references/extreme_shapes.py`.

### Shared gamma steps

Both precisions use the same Lanczos terms, reflection switch, incomplete-gamma series
and continued-fraction recurrences, and Edgeworth correction. Direct tests check the
first terms, iteration budgets and adjacent floats at both crossovers. The complete
series and fraction agree with elementary closed forms at integer shapes.

Reflection reduces the offset from the nearest pole before multiplying by π. When the
reflected positive gamma overflows, binary64 divides by the two Lanczos half powers
separately. Binary32 evaluates this tail in binary64 and rounds on narrowing. MPFR
fixtures cover `Γ(-172.5) ≈ -1.11978350329e-312`, signed underflow and both sides of
the pole at `-2`. The generated reflection laws use a compensated sine reference and
retain their original tolerances.

### Direct tests for Airy and normal branch choices

Fourteen exclusion patterns matched 31 mutation sites in the preceding candidate.
The normal kernels now share log-CDF regime selection, probability routing, seed
selection and Newton refinement across both precisions. Tests observe the decisions
directly at the cutoffs and exercise both a finite central seed and its fallback.

The Airy kernels share the axis and positive-regime decisions. The series uses the
shared strict relative-term check. The scaled `Bi` pair evaluates each even modified
Bessel asymptotic once, and the tests compare both components to the original sum of
opposite orders. Tests around the join still compare the outputs of the two formulas.

The unfiltered workspace catalog also showed that the two `powi_i64` comparison
patterns had no matches after the exponentiation refactor. Removing them leaves 44
exclusion patterns.

### Exact scaling and numerical branch tests

The next pass brought the configuration to 10 patterns. The shared `ldexp` helpers
classify the final exponent and reconstruct both precisions with one rounding at the
subnormal boundary. The regression cases include inputs just above half the least
subnormal, where rounding in two multiplications previously returned zero.

Cutoff tables exercise the integral and zeta routes. The eta series uses exact integer
normalization, and the zeta reflection reduces the angle before evaluating the sine
near trivial zeros. Independent MPFR values cover points on both sides of those zeros.
Direct tests also cover binomial symmetry, interval signed-zero ties, weighted tape
selection, cache capacity, and the runner's panic and score decisions.
Relative tolerance checks scale opposing finite values before subtracting when their
difference overflows. Near-equal values still subtract first to retain the small gap.

The remaining rules cover disjoint bit fields, constructors that return their defaults,
and the direction of the tiny perturbation in `cancelling_pair`.

### Closing the remaining exclusions

The configuration now includes all generated mutations. Assembly of disjoint fields
uses integer addition, the tape hasher reads its zero-padded tail in big-endian order,
and constructors with the same default and identity share one definition. A replay
test fixes the cancellation direction at the upper magnitude bound for both signs.

The magnitude generator shares the exact binary power constructor with binade boundaries.
This restores subnormal draws whose previous power calculation overflowed before taking
the reciprocal. The replay test checks every supported exponent for both signs.

### Timeout diagnostics

The earlier rational-order and digamma timeouts reproduce as named test timeouts under
the Nextest `mutation-replay` profile. The rational comparator checks its floor remainder
in debug builds. Digamma shifts a nonnegative argument at most twenty times. Both known
mutations now produce test failures before the mutation runner's deadline. The full audit
also found a mutation that kept the finite-field exponent at zero forever. Square-and-multiply
now runs for at most the number of bits in its exponent.

The `f32` digamma audit also exposed a wrong sign at input bits `0x3fbb16c3`. Reusing the
`f64` kernel fixes the sign and preserves the small result on either side of the positive
root. The tests compare against 256-bit MPFR references. See the
[replay workflow](CONTRIBUTING.md#diagnose-timeouts) for inspecting test and build timeouts.

### Workspace baseline for 0.6.0

The full run on `43f7096fbbbc` applied 11,688 configured mutations and used the
all-features workspace suite for each mutation.

| Crate | Caught | Unviable | Timeouts | Survivors |
|---|---:|---:|---:|---:|
| `euclid-algebra` | 463 | 101 | 0 | 0 |
| `euclid-matrix` | 21 | 6 | 0 | 14 |
| `euclid-quantity` | 2,680 | 77 | 0 | 0 |
| `euclid-random` | 6 | 17 | 0 | 10 |
| `euclid-scalar` | 6,120 | 65 | 45 | 298 |
| `euclid-stability` | 832 | 396 | 16 | 3 |
| `euclid-tensor` | 44 | 78 | 0 | 18 |
| `euclid-units` | 130 | 63 | 1 | 32 |
| `euclid-utils` | 131 | 0 | 4 | 17 |
| **Total** | **10,427** | **803** | **66** | **392** |

The next pass strengthens the shared test helpers and shrinker. Grid endpoints keep
their exact bits, assertions have rejecting controls, and both the shrinker and runner
advance independently of memo hits and report counters. A replay of 2,000 seeded
shrinker inputs matched the old results and probe accounting. Tensor element counts now
return zero whenever an extent is zero, including after an overflowing prefix product.

The scalar pass separates integer exponentiation from reciprocal handling and trigamma
reflection from its positive recurrence. Private-stage tests check exact binary powers
and independently evaluated trigamma series values. Public regressions check the signed
subnormal powers and both sides of remote poles.
The double-precision beta and upper gamma fractions share one Lentz update.
Unit tests check consecutive Fibonacci convergents, cancellation in either pivot,
and the fractions for integer shapes independently of the CDF prefactors.
The rising-factorial dispatch, Stirling ratio and Bernoulli correction also have separate
checks, so a correction that disappears when rounded into the final result is still tested.
The rising-factorial threshold exclusion was removed. The bounded product uses
one range check, and the Stirling route checks the smaller endpoint.
Quantile tests now exercise bracket expansion, each Newton acceptance condition, and
the Wilson–Hilferty transform directly. Bessel quadrature checks its truncation point
and integrand against independent identities, including a subnormal half-order case.
The isolated Newton tests retain one equivalent comparison in the report. Allowing
zero density cannot produce a finite interior step because the division yields
an infinity or NaN. The gamma seed tests require a positive fallback when the
transform yields exactly zero, since that seed would stall geometric expansion.

## Where it stands

| Crate | Mutants | Caught | Survivors |
|---|---|---|---|
| `euclid-quantity` | 2,764 | 2,687 | **0** † |
| `euclid-units` | 190 | — | **0** † |
| `euclid-tensor` | 140 | — | **0** † |
| `euclid-utils` | 152 | — | **0** † |
| `euclid-random` | 33 | — | **0** † |
| `euclid-matrix` | 19 | — | **0** † |
| `euclid-stability` | 1,186 enumerated | — | measured per file — see below † |
| `euclid-algebra` | 459 | 359 | **0** ‡ |
| — its `conformance.rs`, focused | 53 | 42 | **0** (11 unviable), *measured* |
| `euclid-scalar` | **6,560** | **5,866** | **584** — [the aggregate](#euclid-scalar-measured-whole-for-the-first-time-since-the-merge) |

**† Not re-measured since the restructuring that invalidated `euclid-scalar`'s f32 scores.**
Every one of those crates was last touched by `ed86848` (the `euclid-numeric` merge) and
`euclid-stability` by `68984a5` — and the f32 correction below is what happens when a per-file
score outlives the layout it was taken on. Treat these zeros as *history*, not as current
evidence, until each crate has had one clean run. They are plausible — several were closed to
zero deliberately — but plausible is what the f32 row was too.

**‡ Inferred, not measured.** The crate-wide 459/359 figure predates this work; only its
`conformance.rs` was re-run, and the 13 survivors it held are confirmed closed. The step from
"those 13 are gone" to "the crate has none" assumes nothing else survives, which no run has
checked. One full `-p euclid-algebra` pass would settle it and is cheap at 459 mutants.

`euclid-quantity` was **1,966 survivors of 2,768** when first measured — the worst in the
workspace — and `euclid-stability` was 124.

The pre-refactor stability baseline was 886 mutants, 624 caught and three survivors. Those
numbers are historical, not an aggregate score for the structured runner. Mutation workers
also need fixture isolation: a mutant that saves an invalid choice tape can leave a stale
regression that falsely catches later mutants. The wrapper restores regression and budget
fixtures between scenarios while retaining real persistence in engine tests' scratch paths.
Interrupted quota-limited runs and runs contaminated by those writes are not current evidence.

### Structured runner audit

The isolated audit covers these files. Each row uses the latest complete result for each file;
the full batch and focused follow-up overlap and must not be added together.

| Scope | Mutants | Caught | Timeouts | Unviable | Survivors |
|---|---|---|---|---|---|
| `runner.rs`, `runner/{execute,failure,result}.rs` | 174 | 155 | 3 | 16 | **0** |
| `source.rs`, `stats.rs`, `target.rs` | 123 | 120 | 0 | 3 | **0** |
| `generate/compound.rs` | 65 | 11 | 0 | 54 | **0** |
| `generate/{float,float32}.rs` | 220 | 72 | 0 | 148 | **0** |
| `generate/explore.rs` | 53 | 52 | 0 | 1 | **0** |
| **Total** | **635** | **410** | **3** | **222** | **0** |

This is a scoped measurement, not a new score for all 1,186 stability mutants. The three
timeouts break runner termination by replacing a stop condition or increment. Unviable mutants
failed compilation; many try invalid constructors for `Gen<T>` and are not evidence of test
strength. The final runs used four workers, a 20-second timeout, all workspace features, and
debug symbols disabled to bound scratch build space. Raw reports live outside the checkout.

New controls pin discarded labels/scores, replayed assumptions, exact case/rejection/tape
budgets, full-width integer domains, native float boundaries, nonempty variable lists and
bounded enumeration metadata. Requiring interior values on both sides of zero kills the three
overflow-interpolation survivors that endpoint checks admitted. Shared assertions and panics
must retain their identity during shrinking; inconsistent replays cannot save a counterexample.

Two narrow exclusions are justified in `.cargo/mutants.toml`: OR versus XOR when packing
disjoint `u128` halves, and printing the truthful suffix `0 reused` instead of omitting it.
Removing the panic hook is no longer excluded, since its recorded locations now distinguish
failures. No property result or replay check is disabled during mutation testing.

### `euclid-scalar`, measured whole for the first time since the merge

A complete run, four hours, no interruption:

| | |
|---|---|
| Mutants | **6,560** |
| Caught | **5,866** |
| Survivors | **584** |
| Timeouts | 45 (counted as caught — a mutated loop that spins hangs the suite) |
| Unviable | 65 |

This is the number the table below has been missing, and it **supersedes the per-file "was/now"
figures where the two disagree.** They do disagree, and the largest disagreement is the one worth
reading first.

**The `kernels/f32/**` row claiming 0 survivors is wrong.** The aggregate finds **234** across
the f32 tree — 79 in `f32/gamma.rs` where this table says `91 → 0`, 57 in `f32/quantile.rs`
against `118 → 0`, 37 in `f32/polygamma.rs` against `77 → 0`, plus 30 in `f32/bessel`, 17 in
`f32/repr.rs` and 14 in `f32/normal.rs`. (The `f32/repr.rs` 17 have since been re-measured to
**1** — read the aggregate's per-file figures as the state at the time of that run, not as
current, wherever a later focused run appears below.)

Not an aggregate-versus-focused artifact: a focused run on `f32/gamma.rs` alone returns 435
mutants, 350 caught, **79 missed** — the same number. The `cross_precision` suite credited with
those closures is present and passing, so nothing regressed in the tests. The likeliest reading is
that the closures were measured before the `euclid-numeric` → `euclid-scalar` merge moved the
kernels and their properties, and this file said so at the time: *"a full run remains necessary to
establish one after the merge"*. It was, and the per-file claims did not survive it.

The lesson is about this document rather than the code: **a per-file score is not a crate score,
and a crate that has been restructured has no score at all until it is re-measured.** Rows below
that have not been re-run since the merge should be read as history.

| File | Mutants | Was | Now |
|---|---|---|---|
| `impls/natural_impl.rs` | 159 | 69 | **0** |
| `impls/integer_impl.rs`, `impls/boolean_impl/**` | 258 | 103 | **0** |
| `kernels/host/factorial.rs` | 119 | — | **0** (119 caught) |
| `kernels/f32/**` | 1,239 | ~300 | ~~0~~ **234** — see the correction above |
| — of which `f32/gamma.rs` | 435 | 79 | **43** (386 caught) |
| `kernels/polygamma.rs` + its f32 peer | 262 | 52 | **24** (238 caught) |
| `impls/real_f64_support.rs` + its f32 peer | 61 | 32 | **0** (59 caught, 2 timeouts) |
| `impls/real_f32.rs`, `kernels/erf/f32.rs` | 470 | 191 | **15**, every one characterised below |
| `kernels/transcendental.rs` | 194 | 46 (of 184) | **43** (151 caught) |
| `kernels/gamma.rs` | 443 | 31 | **23** (401 caught, 19 timeouts) |
| `kernels/beta.rs` | 224 | 47 | **19** (204 caught) |
| `kernels/quantile.rs` | 149 | 56 | **56** — equivalent by design, see cause 8 |
| `kernels/bessel/modified_first.rs` + its f32 peer | 569 | 47 | **39** (530 caught) |
| `kernels/repr.rs` + its f32 peer | 166 → 156 | 17 | **2** (154 caught; 10 excluded as proven equivalent) |
| `conformance.rs` | 35 | 13 | **7** (21 caught, 7 unviable) |

The carrier/f32 pass closed **575 survivors**, two of them defects rather than gaps.
The subsequent Lambert W/zeta pass found two more defects and reduced that file's survivor
count from 46 to 43, with no new exclusions. Its population grew from 184 to 194 because the
overflow fix introduces a scaled residual alongside the negative-argument path.

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
  input crossed any of them when this was written. **All four are crossed now**, and the last
  one to be closed is the one that was hiding a defect — see
  [the beta's Edgeworth branch](#the-beta-at-large-shape-a-nan-for-every-x). For the record of
  which instrument closed each, since three of the four were already covered by work done for
  other reasons and only the audit showed it:

  | Threshold | Crossed by |
  |---|---|
  | `GAMMA_EDGEWORTH_MIN` | `identities::gamma_large_shape`, which sweeps the shape to `1e300` |
  | `PSI_ASYMPTOTIC_MIN` | `digamma`/`trigamma` against exact harmonic and zeta sums, `n = 1..200` |
  | `STIRLING_RATIO_MIN` | `rising_factorial_handles_shifts_across_the_stirling_domain` |
  | `BETA_EDGEWORTH_MIN` | `identities::beta_range`, then `identities::beta_large_shape` for the top of the range and for `γ₁` |
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

Found in `euclid-scalar`, and it accounts for the largest single cluster in the crate — the
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

Also `euclid-scalar`, and the counterpart to cause 3's "parallel implementation": the f32 body
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

### 8. A self-correcting iteration, whose inputs cannot reach its output

The largest single cluster still open — **56 in `kernels/quantile.rs`** — and the one that looks
most like a hole and is least like one. It is cause 6's mirror image, which is the reason it gets
its own entry rather than a line in that one.

`gamma_lr_inv` and `beta_reg_inv` are safeguarded Newton: a cheap seed, a bracket expanded until
the root is strictly inside it, then Newton steps that are *rejected and replaced by bisection*
whenever one would leave the bracket or fail to shrink it. The survivors are concentrated in
exactly the inputs that machinery exists to tolerate — twelve in the Wilson–Hilferty `gamma_seed`,
seven in `beta_pdf`, the rest in `gamma_pdf` and the bracket walk.

The docstrings already claimed this ("used only to seed the bracket search, so its accuracy
affects speed rather than the answer"). The claim is now measured, by the standard this file sets
for an equivalence — compute the right answer from an independent source:

| Kernel | Worst relative error vs 120-digit `Decimal` |
|---|---|
| as shipped | `1.1e-13` gamma, `1.3e-14` beta |
| seed disabled **and** both densities forced to `0` | `7.4e-14` gamma, `1.3e-14` beta |

With the seed gone and the derivative identically zero — so no Newton step can ever fire and the
solver degrades to pure bisection — the answers are unchanged, and the gamma case is marginally
*better*. That is not a tolerance being generous; it is the bracket invariant doing what it is
for. A mutation of the seed or the density changes how many iterations are spent and nothing else.

**So do not write assertions against these.** No accuracy test can distinguish them, because
there is no accuracy difference to find. Two things follow, and the second is the useful one:

- They are not excluded in `.cargo/mutants.toml` either, and deliberately. An `exclude_re`
  naming `gamma_seed` or `beta_pdf` would hide *every* future mutation of those functions,
  including the ones that would matter if the solver stopped being safeguarded — and the file's
  own rule is that a pattern must name the operator, never just the function. 56 patterns to
  silence 56 provably-equivalent mutants is a worse trade than a paragraph here.
- **What is observable is the caps**, and they are where the real defects were. `MAX_STEPS` and
  `BRACKET_OCTAVES` convert "a far-off seed" into "a wrong answer" the moment they are too small,
  and both have been wrong: at 200 octaves `gamma_lr_inv(p, 1e-3)` returned `6.2e-64` for every
  `p` from `1e-12` to `0.5`, a constant function of its own argument. `quantiles::extremes` pins
  that, and it is the test that actually protects this kernel.

The generalisation: when an iteration is built to converge regardless of where it starts, its
starting point is unobservable *by construction*, and a mutation score over that region measures
the design rather than the tests. Check whether the fallback is unconditional before reading the
survivors as a gap — and then check the thing that bounds the fallback, because that is where the
answer can actually go wrong.

## Almost none of them was a bug — and then three were

For the first three crates, every law added passed on first run. The instrument was reporting
**where tests were missing**, not where code was broken, and the standing advice was to treat a
survivor as a missing assertion until proven otherwise.

That advice still holds as a prior. But `euclid-scalar` has now produced three exceptions, and
all are worth knowing in detail, because none looked like a bug from the outside and none was
found by a failing test.

All three were found the same way: **not by reading the survivor, but by doing the work to decide
whether the survivor was equivalent.** That work means computing the right answer from an
independent source, and all three times the independent source disagreed with the code.

The third adds a wrinkle the first two did not have. `binomial` and `erf_inv` were wrong where
they were being tested; the beta was wrong in a branch **nothing had ever executed**, and the
survivor count there was not a signal at all — it was 47 mutants in code no test reached, which
reads identically to 47 mutants in code that is merely under-asserted. The distinguishing
question is not "how many survived" but "has this branch ever run", and that is worth asking
first because it is cheaper to answer.

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

### The unreached branch: the beta's Edgeworth normal

The third is the one above, and the conclusion is short: `beta_reg` returned **NaN for every `x`**
whenever `a + b` exceeded about `5.6e102`, because `n·n·(n+1)` overflowed while the standard
deviation it encodes was `3.5e-76` and the answer — `I_½(a,a)` — was exactly `½`. Two further
intermediates, `√(a·b)` and the variance at an extreme ratio, failed the same way.

It is the Lambert W defect in a different function: an intermediate leaving the format while the
result never does. Both were found by pushing a kernel to the edge of its *representable* domain
rather than its documented one, which is the sweep worth repeating anywhere a formula multiplies
three `O(n)` factors together.

What made it survive is the part worth generalising, and it is not "the branch was untested" —
the branch was tested, by `beta_range`, on an exact identity, from `1e7` to `1e100`. It survived
because **the sweep stopped at a round number two and a half decades short of the defect.**
`1e100` is not a boundary of anything; `f64` runs to `1.8e308`. A sweep that ends on a power of
ten looks complete in a way that a sweep ending at `f64::MAX` actually is, and the difference
here was 206 decades of shape and one `NaN` for every argument in them.

The gamma peer makes the contrast exact: `gamma_large_shape` sweeps to `1e300` and has no
equivalent hole. Same template, same author, different last element of the array.

### What to carry forward

Still treat a survivor as a missing assertion first; that is right the great majority of the
time. But six refinements, all paid for:

1. **When deciding whether a mutant is equivalent, compute the right answer from an independent
   source.** That step is what found all three bugs. Reasoning from the source found none of them
   — in the `binomial` case the source's reasoning was itself the error.
2. **A large cluster in one small function is a design signal.** Twelve wrong coefficients, or
   thirty-four survivors in forty lines, says the function's internals do not reach its output.
   Ask why before writing a test.
3. **Check whether the branch executes before theorising about why its mutants survive.** It is
   a one-line experiment — put a `panic!()` in it, run the suite — and it separates "no test
   reaches this" from "tests reach it without pinning it". This file asserted the first about the
   beta's Edgeworth branch and the experiment said the second: two `beta_range` tests had been
   running it all along. Run the check; the answer is cheap and it was not the expected one.
4. **End a magnitude sweep at the format's limit, not at a round number.** `1e100` and `1e300`
   both look like "as large as anyone cares about". One of them left 206 decades untested and a
   `NaN` in all of them. Where a sweep exists to establish that a kernel survives its whole
   domain, the last element should be `f64::MAX` or the value that provably saturates — and if
   that is too slow, the comment should say where it stopped and why.
5. **Assert a guard over two arguments at each argument separately.** `next_after`'s
   `x.is_nan() || y.is_nan()` was weakenable to `&&` under a test that passed NaN as the *first*
   argument, because that argument's fall-through path walks a NaN's bits and lands on another
   NaN — the right answer by accident. The second argument's path returns a finite neighbour.
   A guard that reads symmetrically is not symmetric when one side's failure reconstructs the
   answer, and the same goes for the `||`-to-`&&` weakening generally: ask which *operand* the
   existing test exercises. See
   [the representation pair](#the-representation-pair-17-survivors-and-a-guard-tested-from-only-one-side).
6. **A generated property is only as strong as its generator's reach, and a false law can hide
   behind a generator that cannot reach it.** `ulp_agrees_with_next_after` asserted
   `ulp(x) == |next_after(x, +∞) − x|` for *every* `x` — which for a negative argument is the
   step toward zero, the opposite of the documented contract. It passed, and would have gone on
   passing, because the two directions differ only on a binade boundary and
   `generate::float::magnitude` fills all 52 mantissa bits by design, making that a `2^-52`
   event. So the suite held a wrong law, a real defect of exactly that shape went unnoticed, and
   the fix was a **generator** (`binade_boundary`) rather than a property. The diagnostic
   question for any property that looks like it should have caught something: *can its generator
   produce the input that distinguishes the law?* Both halves of that pair now assert the same
   corrected law, and only the one drawing boundaries detects the defect — which is the cleanest
   available demonstration that the generator, not the assertion, was the missing piece.

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

## `euclid-scalar` — the carrier tiers and the f32 tree

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


### Lambert W and zeta: two defects at representable extremes

The baseline was 184 mutants: 138 caught and 46 survivors. The updated kernel is 194 mutants:
151 caught and 43 survivors, with no timeouts or unviable mutants in either run. This is a
file-scoped measurement, not a new total for `euclid-scalar`.

**Lambert W near `f64::MAX`.** The answer is only about 703, but `w*exp(w)` and
`f*(w+2)` in the Halley step can overflow. Refinement then silently stops early: the residual
in `w + ln(w) = ln(x)` was `4.33e-5` at the largest finite input. Dividing the residual and
denominators by `exp(w)` for positive arguments removes both overflowing products. The
negative branch retains its original residual to avoid changing cancellation near `-1/e`.
The regression checks the logarithmic identity up to `MAX` and its predecessor, with a
measured residual of at most one ulp of `ln(x)`.

**Zeta immediately left of zero.** For a tiny negative argument, `1-s` rounds to exactly
`1`. Reflection therefore evaluates the Hurwitz-zeta pole: a debug assertion fails at a
regular point where zeta should be almost `-1/2`. The accelerated eta calculation now covers
`-1 < s < 1`, avoiding that subtraction near zero. Tests check both signs down to the
smallest subnormal against `zeta(0) = -1/2` and `zeta'(0) = -ln(2*pi)/2`, independently
Euler-transformed eta references at negative quarters, and the exact Bernoulli value at
the eta/reflection seam.

The branch-point tests also require `W >= -1` without tolerance: that is the principal
branch's domain, not an accuracy estimate. Independent 90-digit Decimal bisection values
check the root itself, because substituting into `w*exp(w)` cannot resolve it near the
vanishing derivative.

The **43 remaining survivors** split into 35 in Lambert W (initial guesses, iteration
corrections and fallback paths), two in eta's normalization, and six in zeta's guards and
reflection sign logic. Refinement can recover from a changed guess or correction, so a
survivor there is not automatically a missing accuracy case. These are left visible for
further investigation; this pass does not establish that they are all equivalent.

Raw baseline and updated reports were written outside the repository with `--output`;
`mutants.out` and `mutants.out.old` remain ignored. Keep the diagnosis here, not generated logs.

### Gamma: eight missing assertions closed by generated domains

The gamma-family baseline was 443 mutants: 393 caught, 19 timeouts and 31 survivors.
The same population with `tests/properties/gamma.rs` is 401 caught, 19 timeouts and 23 survivors.
All four new properties also pass a 25x soak. No production code or exclusions changed.

Six weakened incomplete-gamma guards survived because arithmetic on NaN usually returns NaN
anyway. Pairing a NaN or negative shape with an infinite argument, or a valid shape with negative
infinity, distinguishes the guard from the infinity shortcut that follows it. Both tails now
reject those mixed cases.

A trigamma guard could reject every negative non-integer without failing the fixed suites.
Generated recurrence checks between its negative poles now distinguish poles from ordinary
negative arguments. The remaining closure is the differenced-Stirling domain: both endpoints
must be large. Anchoring one endpoint at `Gamma(1/2) = sqrt(pi)` and generating the other above
the threshold, in both directions, catches changing that conjunction to a disjunction.

The eight closed sites were rechecked after moving the kernel and properties to `euclid-scalar`;
the focused run also includes the two small-integer conjunction mutations, whose loops time out.

The 23 survivors remain visible. They include boundary comparisons, continued-fraction tiny
guards, non-finite rising-factorial fallbacks and Stirling correction arithmetic; this run does
not classify all of them as equivalent.

### Still untouched

Nothing, as a *cluster*. All three rows that used to sit here have been measured, and they came
out in three different places — one hiding a defect, one hiding nothing, one hiding dead code.
The sections below are the results. What remains open is narrower and listed under
[what is left](#what-is-left).

#### Modified Bessel `I₀`/`I₁`: fifteen mutants in a branch that cannot run

Listed here as "41, —", with no diagnosis. A complete run found **290 mutants, 254 caught, 36
survivors**, and fifteen of those survivors were on three consecutive lines of
`ln_i1_minus_x` — every operator in one `ln_peak` expression.

A cluster that size in one small function is the design signal this file records twice over, and
the answer was the cheapest possible one: **the branch is unreachable.** `k_peak` is
`round(ax/2)`, below 1 exactly when `ax < 1`, and both callers route `ax < 1` to `i1_no_log`
instead — a path added precisely because the logarithm round trip is lossy for small arguments.
The `I1_NO_LOG_MAX` docstring had even stated the condition (`k_peak = round(x/2)` is zero exactly
when `x < 1`) without anyone noticing it made the arm below dead.

Confirmed by the one-line experiment this file now recommends first: a `panic!()` in the arm,
then the suite, then a dense sweep of 20,700 arguments at both precisions, both signs, across all
four entry points. It never fired. The f32 peer carried the same dead arm.

Deleted, and replaced with `debug_assert!(ax >= I1_NO_LOG_MAX)` so the precondition is enforced
rather than merely true — a future caller reaching for a small argument now fails loudly instead
of silently taking a path that no longer exists. **268 mutants, 251 caught, 17 survivors**: the
deletion removed 22 mutants along with the code and killed 19, which is the same bargain as the
f32 `erf_inv` Newton step.

**And then `i1_over_i0` gave up eleven more, to the oldest trick in the file.** `I₁/I₀` is odd,
and it carries the sign in **three** places — once in the peak-factored series, once in the
asymptotic form, once at infinity. Exactly one of them was tested: `i1_over_i0(-3.0)`, which
reaches the series. The other two had never been handed a negative argument, so dropping a
negation or testing `x > 0` in place of `x < 0` went unnoticed — in a function whose entire range
is `(−1, 1)`, that is as wrong as an answer gets.

A sign test written per branch needs a negative argument *per branch*, and "the function is odd,
checked once" is not that. Closed in both precisions across all three branches plus both zeros.

Combined across the pair: **569 mutants, 530 caught, 39 survivors** (from 47). What is left is
the convergence-guard family — eight each in `i0_rel_sum` and `i1_rel_sum` on
`ratio < sum * TERM_EPS`, six apiece in `i1`/`i1e` branch selection, and the `i1_over_i0`
remainder, which are the `<=` relaxations that can only differ at `x == 0` where the function has
already returned. Not individually established as equivalent, but no longer concealing a sign.

#### `euclid-scalar`'s own conformance was vacuous too

Found by the aggregate run rather than by looking: thirteen survivors in
`crates/euclid-scalar/src/conformance.rs`, the identical pattern to
[`euclid-algebra`'s](#euclid-algebra--13-now-0). `boolean_laws`, `mask_laws`, `real_laws`, their
three `*_for` wrappers, their three `generated::*` peers, `generated::settings` and the defaulted
`ConformanceSample::tolerance` could each be replaced with an empty body — or, for `tolerance`,
with `-1.0` — while the workspace stayed green.

`tolerance -> -1.0` is the one worth pausing on. A *negative* relative tolerance makes every
inexact bound negative and so every inexact law unsatisfiable; it is not a subtle weakening at
all. It survived because `f64` and `f32` both override the default, so the defaulted body never
ran — the same unreachable-default shape as the algebra tower's `cases`.

Closed to **7 survivors** by `src/tests/vacuity.rs`: an `and` wired to `or`, a `le` wired to
`lt`, a `select` with its branches swapped (the defect `mask_laws`' own docs say the select laws
exist for), and direct assertions on the two defaulted knobs. `settings` is `pub(crate)` for the
same reason as its algebra peer.

One lesson from doing it: `mask_laws_for` was missed on the first pass and only the
re-measurement caught it. The three `*_for` wrappers look alike, two had rejections written, and
reading the file did not reveal which. Measure rather than reason about which wrappers are
covered.

#### The f32 rising factorial: a branch five times short of the sweep

The largest single cluster the aggregate turned up: **45 of `f32/gamma.rs`'s 79 survivors in one
function**, `ln_rising_factorial`. Investigated the way this file now recommends — `panic!()` in
the branch, then a full-workspace run — and the answer was that the integrated-digamma
asymptotic, taken when `a > 1e2 && a + n > 1e2`, **was executed by zero tests**.

Two causes, and both are repeats of mistakes recorded elsewhere in this file:

- **The sweep stopped short of the threshold.** `cross_precision` exercises this kernel over
  `a ∈ [0.5, 20]`, and the f32 threshold is `100`. Five times short, and the same shape as the
  beta's Edgeworth defect — a branch the suite reaches *near* but never *into*.
- **The budget was loose for a documented reason, and loose is loose.** That sweep allows 301 ulp
  at `n = 0.5`, honestly, because the function crosses zero near `a = 1.23` where ulp spacing
  collapses while the derivative does not. It is also wide enough to admit almost any mutation
  of the arithmetic.

Unlike the Bessel case this branch is **not** dead code — `Real::ln_rising_factorial(200.0f32,
0.5)` reaches it from the public API — so the fix is a test, not a deletion. Above the threshold
neither excuse applies: the result is large and well-conditioned, so both available instruments
are tight, and both were measured before a bound was written:

| Instrument | Measured worst | Bound |
|---|---|---|
| f32 against the correctly-rounded f64 | 2 ulp at every `n` tried | 4 |
| `ln (a)ₙ = Σ ln(a+k)`, integer `n` | 1.72e-7 relative (≈1.4 f32 ulp) | 4e-7 |

The second matters more: it checks against an *exact identity* rather than euclid's own f64 path,
so it still fails if both precisions are wrong together. `n` is taken above 64 so the
exact-product regime is not what answers.

Result: `ln_rising_factorial` **45 → 9**, and the file **79 → 43**.

#### The `NormedDivisionAlgebra` tier: 32 survivors in a surface nothing instantiated

The two `*_support.rs` files were never mentioned in this document before the aggregate run, and
they held **32 survivors between them** — sixteen each, in matched pairs. Every one of the five
`NormedDivisionAlgebra` methods (`conjugate`, `norm_sqr`, `real_part`, `from_real`, `magnitude`)
was replaceable with a constant, in both precisions.

Cause 3, in a form worth naming separately: **an inherited tier**. `Real` is
`NormedDivisionAlgebra<RealPart = Self> + Compare`, so the float carriers pick these five up
without writing them into the block the suites exercise. `real_laws` does not mention any of
them — zero occurrences in `conformance.rs` — and no test in the workspace called them either.
The tier above the one being tested is invisible, and it is invisible in *pairs*, because each
precision has its own impl block.

The same finding as `euclid-algebra`'s `f32_satisfies_the_tower_it_claims`, which records
`div_rem` and `conjugate` replaceable with `Default::default()` for exactly this reason.

Closed by `tests/real_carriers/normed_algebra.rs`, and the laws relate operations rather than
restating bodies:

- `norm_sqr(x) = x · conjugate(x)` — the defining relation, checked against `*`
- `|x|² = norm_sqr(x)` — ties the two norms together
- **`|x·y| = |x|·|y|`** — multiplicativity, which is what *normed* means in the trait's name, and
  the one law a constant-returning `magnitude` cannot satisfy
- `from_real(real_part(x)) = x`, `|−x| = |x|`, `|x| ≥ 0` and zero only at zero

Two of the five are the identity on the real line, and asserting that is not vacuous: the trait
promises it, a generic algorithm depends on it, and `conjugate` in particular is what makes an
adjoint or Hermitian routine correct over ℂ as well as ℝ — a wrong one is silently real-only.

A separate test records an asymmetry worth having written down: `magnitude` is exact at every
finite argument while `norm_sqr` overflows at `f64::MAX` and underflows at `5e-324`, because one
squares and the other does not.

`Interpolate` for `f32` went the same way — three more survivors, `midpoint` replaceable with
`Default::default()` and `lerp`'s subtraction with an addition, because the existing test covers
`f64` only and the f32 impl is its own block.

**32 → 0.** Two of the final survivors were excluded rather than tested, and one of those
exclusions was *already planned*: the file said the float `zero()` was equivalent "by the same
argument" as the integer carriers' but that `impls/real_f{32,64}.rs` had not been measured yet,
and "an exclusion written ahead of the measurement is how a real survivor gets hidden". They have
now been measured, so it is written. `powi_i64`'s two boundary relaxations are excluded on a
one-line reachability argument each, and the run confirms sixteen other `powi_i64` mutations are
still caught — which is the check that says the patterns did not hide anything.

#### The f32 beta quantile stopped on an absolute bound — and fixing it *raised* the survivor count

The largest remaining f32 cluster was 57 in `f32/quantile.rs`, and the standing guess in the
table below was that it would be [cause 8](#8-a-self-correcting-iteration-whose-inputs-cannot-reach-its-output)
again — the f32 peer of a solver whose internals cannot reach its output. The guess was half
right in a way worth recording, because the other half was a defect.

**`beta_reg_inv` could not resolve a root below `5e-7`.** Its loop exited on
`(hi - lo) <= 4.0 * f32::EPSILON` — an **absolute** bound of `4.8e-7` — so for any root beneath
that, the bracket satisfied the test within a few steps and the *midpoint* came back instead of
the root. `I_x(0.1, 0.1) = 0.05` returned `2.38e-7`, which is `2⁻²²` and visibly the bisection
halting, against a true root of `8.66e-11`: **2,752 times too large.**

Three things made it findable, and all three are techniques this file already carries:

- The f32 kernel was compared against the **f64 kernel**, four orders finer — the instrument
  cause 3 and the cross-precision suites are built on.
- Only roots representable as a normal `f32` were compared. Without that filter the small-shape
  gamma cases report a relative error of `5.7e278`, which is meaningless noise: at `a = 1e-3` the
  true root is `p^1000 ≈ 1e-602` and simply underflows. Filtering is what left one real signal.
- The *sibling* was right. `gamma_lr_inv`, three screens up in the same file, scales by
  `hi.abs().max(f32::MIN_POSITIVE)`, and so does the f64 peer whose documentation explains why:
  "iterating until the bracket stops shrinking, rather than until a fixed tolerance, is what
  makes this parameter-independent". One of three implementations did not scale, and the
  inconsistency is what identifies the defect as a defect rather than a choice.

Worst relative error against `f64` over U-shaped shapes: **`2.8e3` → `3.0e-7`**, about 2.5 `f32`
ulp. `f64` was never affected. Nothing above `~5e-7` moves.

**Then the survivor count went up: 57 → 73.** That is not a regression, and understanding why
matters more than the number. The increase is entirely in the two things the fix touched —
`beta_reg_inv` 18 → 27 and `beta_pdf` 1 → 8 — while `gamma_lr_inv` (31) and `gamma_start` (7) are
untouched and unchanged.

Before the fix the loop stopped early, so *the path mattered*: pure bisection halted somewhere
different from a Newton-assisted descent, and perturbing the seed or the density changed the
answer. That made the mutants observable **for the wrong reason** — they were caught by a bug.
Afterwards the loop runs until the bracket is relatively tight and converges to the root
regardless of how it got there, which is the whole point of a safeguarded solver. Verified the
same way as the f64 peer:

| | worst relative error vs the `f64` kernel |
|---|---|
| as shipped | `2.984e-7` |
| density forced to `0` (no Newton step possible) **and** seed replaced by a constant `0.5` | `3.052e-7` |

Identical to within a quarter of an ulp. So `f32/quantile.rs` now joins `kernels/quantile.rs` as
genuine cause-8 territory, and the sixteen new survivors are evidence the fix took.

**The lesson is about reading the instrument.** A rising mutation score can mean the code got
*more* correct, because a self-correcting algorithm has unobservable internals by construction
and a broken one leaks them. Treat a cluster that grows after a fix as a hypothesis about the
fix, not a verdict on it — and check whether the thing that was being "caught" was the bug.

#### A constant that means something else in the narrower type

Found in the same file while reading it. `gamma_start`'s fallback floored its result with
`.max(1e-300)`, carried over from the f64 peer — and **`1e-300` underflows to `0.0` in `f32`**,
so the floor floored nothing. Harmless, because the caller applies `.max(f32::MIN_POSITIVE)`
anyway, but it is the same hazard as `exprel`'s dead Taylor terms in a different disguise: a
literal that is meaningful at one precision and silently vacuous at the other. Now written as
`f32::MIN_POSITIVE`, which says what it means in the format it is in.

#### `exprel`: dead precision in both precisions, by different amounts

The second-largest f32 cluster: **23 of `f32/polygamma.rs`'s 37 survivors in `exprel`**, and all
23 on three lines. `(eˣ−1)/x` takes a Taylor branch below `|x| < 1e-5`, carrying
`1 + x/2 + x²/6 + x³/24` — the f64 body with the type changed, which is cause 7's exact
signature.

Measured rather than argued, over 400,000 arguments either side of zero:

| | f32 (ε = 1.19e-7) | f64 (ε = 2.22e-16) |
|---|---|---|
| dropping `x³/24` | 0 results change | 0 results change |
| dropping `x²/6` as well | 0 results change | **798,238 change** |
| terms that earn their place | **2** | **3** |

So both precisions carried dead arithmetic, and *different amounts of it*. At f32 the `x²/6`
term is `1.7e-11` against a resolution of `1.19e-7` — four orders too fine to see — while at f64
the same term is five orders *above* resolution and essential. The two terms look alike and are
not: **the cutoff is the format's resolution, so it has to be recomputed per precision rather
than inherited with the body.** That is the generalisation worth keeping, and it is why
narrowing an f64 kernel by changing the type is never finished.

The branch itself survives the audit, which was worth checking rather than assuming: against the
truth `1 + x/2` is accurate to `5.96e-8` — half an f32 ulp, correctly rounded — while
`exp_m1(x)/x` reaches `1.07e-7`, and the two disagree on 106,786 of the sampled arguments. So the
series is a real improvement, not defensive habit, and
`exprel_is_correctly_rounded_below_its_crossover` pins the `7e-8` bound that separates them —
which is also what kills the mutant that would narrow the crossover until the series stops being
used.

**One survivor is deliberately left open.** The same crossover mutation at *f64* needs a reference
finer than `f64` to see: the f32 test can use `f64` as its truth, and nothing in-tree plays that
role for the wider type. The honest fix is a pasted table from `bin/references/`, not a loosened
tolerance, and the kernel says so where a reader will find it.

#### `rgamma`'s two infinities, and four guards that cannot differ

Six mutants — three in each precision — sat on `rgamma`'s `if x > 0.0` inside its non-finite arm,
because **no test ever passed `rgamma` a non-finite argument**. The arms differ for a
mathematical reason: `Γ` grows without bound as `x → +∞` so `1/Γ → 0`, while in the negative
direction `Γ` alternates sign across every pole and has no limit, so `NaN` is the honest answer.
Swapping the comparison reports a definite zero where no limit exists, which is the worse
direction — a caller sees a plausible number. Closed by
`rgamma_distinguishes_the_two_infinities`, both precisions.

The remaining four in that function are **unreachable rather than unchecked**, and are worth
naming so nobody writes a carrier for them: `x > 0.0` relaxed to `>=` can only differ at `x == 0`,
which the pole guard above it has already returned for; and `s < 0.0` relaxed to `<=` can only
differ when `sin(πx)` is exactly zero, which happens only at integer `x` — also already returned.

Combined, the two polygamma files went from **52 survivors to 24** (262 mutants, 238 caught).

#### Coverage requirements, applied where the rule now says to

The companion change, and the reason the f64 peer did not have this problem. `ln_rising_factorial`
has three regimes and the generated property labelled only the first cutoff — `n <= 64` against
`n > 64` — which hides the second, since above 64 the kernel still chooses between the asymptotic
and a direct log-gamma difference on `a > STIRLING_RATIO_MIN`. Both arms came up, and nothing
*required* either, so a generator narrowed later would have stopped testing the asymptotic
silently.

Now labelled per regime and all three required, at roughly half their measured shares (81.00%
exact, 14.75% asymptotic, 4.25% direct over 400 cases). The requirement was confirmed to bite by
raising one bound to 90% and watching the property fail with
`required 90.00% of cases, got 4.25%` — a coverage bound that cannot fail is as vacuous as a law
that asserts nothing.

The direct arm is thin (4.25%) because it needs `n > 64` *and* `a <= 300`, which is 6% at best
from the current generators. Widening it would change a generator's range, hence its tape layout,
hence the persisted regression beside the file — a worse trade than a 2% floor on a branch the
fixed suites also cover.

#### The representation pair: 17 survivors, and a guard tested from only one side

`kernels/repr.rs` and `kernels/f32/repr.rs` were the "never diagnosed" half of the `~68` row
below — `frexp`/`ldexp`/`next_after`/`ulp`, the bit-level primitives everything measuring an
error eventually calls. Measured together: **166 mutants, 149 caught, 17 survivors**. They are
now **156 mutants, 154 caught, 2 survivors**, and both of those are the `ldexp` guard this file
already records as deliberately visible.

The 17 were almost perfectly symmetric across the two precisions, which is itself the finding:
these are separate implementations (8-bit exponent against 11) and a hole in one is a hole in
both. Five were real and are closed; ten were equivalent and are now excluded with a committed
search behind them.

**A guard asserted from one argument only.** The best of the five, and a shape worth carrying
forward. `next_after` opens `if x.is_nan() || y.is_nan()`, and weakening that `||` to `&&`
survived in both precisions despite a test that looks exactly like the one that would catch it:

```rust
assert!(Real::next_after(f64::NAN, 1.0).is_nan());   // passes with the mutant
assert!(Real::next_after(1.0f64, f64::NAN).is_nan()); // the one that was missing
```

A NaN *source* cannot show the defect, because the fall-through path walks `x`'s bits by one and
a NaN's neighbour is another NaN — so `is_nan()` holds either way. A NaN *target* falls through
to the direction test, where every comparison against NaN is false, and `next_after(1.0, NaN)`
returns `1.0 − 1ulp`: a plausible neighbour, for a direction nobody asked for. In a bracketing
search that is a step taken on a poisoned bound, with nothing non-finite left to notice.

The general rule: **a guard over two arguments needs the test at each argument separately.** The
symmetric-looking case is not symmetric when one side's failure path happens to reconstruct the
right answer.

**A direction only one input can demonstrate.** `ulp` picks its side with
`if x < 0.0 { NEG_INFINITY } else { INFINITY }` — the docstring's "away from zero". Replacing
that with `x == 0.0`, so every negative argument is measured *toward* zero, survived both
precisions. The two directions have the same width everywhere except at a binade boundary, where
the one below is half as wide, and the sweep used `-3.0` — interior to its binade, and unable to
tell them apart. `-2.0` is off by exactly a factor of two. Same shape as the `1e100` round number
in the beta sweep: the input that would have shown it was adjacent to the ones chosen.

**A guard path nothing entered.** The f32 `frexp`'s `if x == 0.0 || !x.is_finite()` had no
coverage, because the f32 surface check decomposed `1.0` and `2.0` and never passed a zero. Under
`&&` the condition is unsatisfiable and zero takes the subnormal path, which rescales zero and
asks again — **unbounded recursion**, caught by nothing because nothing called it. The f64 peer
asserts this in `frexp_gives_the_expected_decompositions`; this is cause 1 in its usual form.

Thirteen more mutants in `ldexp` were the same cause — the guard pairs `frexp` with `ldexp` on a
finite non-zero value in every existing test, so `x * 2^0` could become `x + 2^0` unnoticed — plus
chunk boundaries never hit exactly. A signed zero catches the first (`-0.0 + 1.0` is `1.0`, while
the multiply keeps the sign bit the round trip depends on); testing `n` *at* the chunk separates
`>` from `==`. That work also corrected a comment: the f32 chunk is 60, not the 512 the f64 peer
uses and the comment claimed, because `f32`'s exponent reaches only 127 and 512 would overflow on
the first step.

**And a contract gap, recorded rather than silently fixed.** `Real::ldexp` documents itself as
`self · 2^exp`. For a **non-finite `exp`** the formula says `+∞`, `0` and `NaN`; the
implementation returns `self` unchanged. The tests pin the behaviour that exists — it is
self-consistent and defensive, and `exp` never arrives non-finite from `frexp`, whose `e` is
always a small integer — but `0 · 2^∞` is an indeterminate form that wants a decision rather than
a guess, and the doc and the code do not currently agree.

#### Lambert W and zeta: the best law in the crate, and a seed that defeats three instruments

`kernels/transcendental.rs` had no generated property and measured **43 survivors of 194**, 35 of
them inside `lambert_w`. Nine properties later it is **36 of 197**, and the seven that died are
worth separating from the twenty-eight that did not.

**`W` has the strongest law available anywhere here: its own definition.** `W` *is* the inverse of
`w ↦ w e^w`, so `W(x)·e^{W(x)} = x` holds at every argument in the domain and needs no reference
value. Checked in two forms, each picked to dodge an overflow rather than a tolerance — in logs
(`ln W + W = ln x`) on the positive axis, where the direct form needs `e^{684}` at `x = 1e300`,
and directly on `[−1/e, 0)` where `|x| ≤ 1/e`. **One ulp in both**, across three hundred decades.

That killed seven, and all seven are in the **branch-point series**. The contrast with the normal
family is the interesting part: there, a property suite over the same shape of problem killed
*zero*. The difference is that Halley's derivative vanishes at the branch point, so convergence is
slow exactly where the seed matters — a bad guess there survives into the answer, where in the
quantile's central region any guess converges.

**The remaining twenty-eight resist three separate instruments, and the measurements say why.**
Sixteen are in the seed (now extracted as `lambert_seed`, so the report attributes them by name)
and twelve in the Halley step.

- *Accuracy of the seed* — the approach that worked for `tail_start` — does not apply, because the
  seed is **genuinely poor and does not need to be good**. The Taylor branch returns `34.5` at
  `x = 3`, where `W(3)` is `1.05`. A test demanding closeness would fail against the real kernel.
- *Cost* does not discriminate either. Counting Halley steps until `w` lands within four ulps of
  the converged answer, the real seed needs **22 at `x = 3`** while four mutated variants need
  eleven, twenty-two, twenty-two and twenty-two. The real seed is not better by the measure that
  would have to catch them.
- *Convergence detection* is unusable as a signal at all: the iteration oscillates at the last
  ulp, so `next == w` never fires and **every** argument consumes the full `W_STEPS = 60`. That
  also explains why the budget is 60.

So these are unreachable from outside, and the diagnosis points at the kernel rather than at the
tests. **The Taylor regime is carried to `x = 3`, far past where `x − x² + 3x³/2` is useful, and
the large-`x` form is already better by `x ≈ 1`.** Narrowing it would save about twenty iterations
per call *and* make the seed observable — the cause-6 reading rather than the cause-8 one, and the
first time in this file that a cost measurement has pointed at a design choice. Not done here,
because it is a kernel change on cost grounds and wants its own decision.

The zeta side's eight are now **all excluded on evidence**. Six are strict comparisons whose
equality case an earlier `return` has already taken — `s > 0.0` inside `if s.is_infinite()`, where
a zero is not infinite; `s > 1.0` after the pole returns; `s < 0.0` after `ζ(0)` returns; and the
`sin_term`/`tail` sign tests, where `tail` is `ζ(1−s)` with `1−s ≥ 2` and so never below one. The
other two are the CVZ divisor's **absorbed** reciprocal: `d = (t + 1/t)/2` with
`t = (3+√8)²⁴ ≈ 2.4e18`, where one ulp is `512`, so `1/t ≈ 4.1e-19` cannot change a bit — flipping
its sign subtracts what was never added, and `1.0 % t` is `1.0`, still thirty-eight orders below
the spacing. Both tests assert the causing inequalities as well as the bit equality, so reducing
`CVZ_TERMS` far enough to matter breaks the test before it breaks the exclusion.

One correction worth keeping: the `tail` claim is asserted as `≥ 1`, not `> 1`. `ζ` saturates to
exactly one once `1 − s` passes about 53 — `ζ(101)` is `1.0` on the nose — and a first draft
asserting the strict inequality failed against a correct kernel. That is the same saturation the
zeta properties record, met from the other direction. Its laws went
in alongside: the Dirichlet series as an **independent** reference for `s ≥ 8` (where the tail past
`k = 200` is `1.1e-17`, against `7.7e-12` at `s = 3` — so the range starts at 8 so the truncation
is not the thing being measured), the functional equation **inside the strip** where the kernel
does not use it, and the closed forms, which are bit-exact at `ζ(2)`, `ζ(0)`, `ζ(−1)` and `ζ(−3)`.

Two traps worth repeating, both caught by measuring. `ln x` is zero at `x = 1`, so the log-form
identity needs an absolute term — **the third suite in a row** to hit a reference crossing zero.
And `ζ(4)` is one ulp from `π⁴/90`, not bit-exact: an exploration printed the two as the same
number, because `Debug` emits the shortest string that round-trips and two neighbours print
identically. Comparing the printed form is not comparing the value.

#### The gamma family: eighteen laws, and the lesson that a true law can test nothing

The counterpart to the Airy result, and more instructive for being a near-failure. The same
instrument — functional equations, reference-free, each constraining a function at two or three
arguments — was applied to `gamma.rs`, `f32/gamma.rs`, `polygamma.rs` and `f32/polygamma.rs`:
**1,140 mutants, 89 survivors**, against the 90 these four files held before. Eighteen properties
later they stand at **83**, and almost none of that came from the laws.

The laws themselves are sound and several are exceptional. Measured before being written:

| Law | Measured |
|---|---|
| `(a)_{n+1} = (a)_n(a+n)` in logs | **0 ulp**, `a` to `1e12`, in f32 too |
| `ln (a)ₙ = Σ_{k<n} ln(a+k)` | **0 ulp** |
| `P(a,x) + Q(a,x) = 1` | 0–1 ulp, `a` to `1e8` |
| `Γ(x)Γ(1−x) = π/sin πx`, negative axis | `2.8e-16` |
| `lnΓ(2x)` duplication to `x = 1e5` | `5.2e-16` |
| `Q(n,x) = e^{−x}Σ x^k/k!` | `1.1e-14` |

And they killed essentially nothing. **Three reasons, and all three are about the test rather
than the kernel:**

- **A bit-exact law was testing one branch of four.** `ln_rising_factorial` has four regimes:
  `n == 0`, a direct `Σ ln(a+k)` for whole `n ≤ 64`, a Stirling expansion for large `a`, and the
  `lnΓ` difference otherwise. The recurrence property drew `n` from `generate::int`, so `n` was
  always whole and always took the direct sum — which satisfies the recurrence *by construction*,
  one more term being exactly one more `ln`. The law was true, exact, and vacuous. This is the
  `ulp` finding in a third costume: **ask which branch a law exercises, not whether it holds.**
- **Five survivors sat in a guard every property skipped on purpose.** The non-finite branch is
  only reached when an argument is `±∞` or `NaN`, and every property in the suite opens with an
  `is_finite` guard. A whole branch had no coverage because the suite's own hygiene excluded it.
- **A hedged assertion is not an assertion.** The first attempt at that branch checked
  `!result.is_finite()`, and killed nothing: the mutations turn one non-finite value into
  another, so a test that accepts any of them accepts all of them. Pinning the *exact* answers —
  NaN for a non-finite shape, `+∞` for an infinite count, `0` for the empty product whatever the
  shape — killed six.

**Since triaged: fourteen down to seven.** One was a real defect, six are excluded on evidence,
and the hunt corrected two of my own wrong diagnoses along the way.

The defect is a **negative count**. `ln (a)ₙ` is defined for any `n` with `a + n > 0`, so `n` may
be negative — and every property in the suite generated `n > 0`. Mutating the Stirling guard's
`a + n > STIRLING_RATIO_MIN` to `a * n > …` therefore survived: for positive `n` the two agree
about the branch, and for negative `n` the product is negative, so the mutant falls through to the
direct difference. At `a = 1e12, n = −3.5` that returns `−96.7109375` where the answer is
`−96.70857390574204` — wrong by `2.4e-5`, five digits gone to exactly the cancellation the
Stirling branch exists to avoid.

The *reference* that catches it is worth naming, because the obvious instrument fails. The
recurrence `(a)_{n+1} = (a)_n(a+n)` holds for the naive form **too** — both `lnΓ` terms carry
nearly the same error, which cancels in the difference — so a self-referential law is satisfied by
a badly wrong value, the same blind spot the seed recurrences had. What works is that for negative
*integer* `n` the answer is an exact, well-conditioned sum: `ln (a)_{−m} = −Σ_{k=1}^{m} ln(a − k)`.
No cancellation, no Stirling, no `lnΓ`.

Six more are excluded: all three `>` comparisons in each precision. `n > 0.0` is unreachable
because `n == 0` returns above it; the two threshold comparisons move which branch evaluates the
threshold exactly, and the function is continuous across the join to `1e-13` over an ulp-sized
step while the Stirling branch matches the independent sum at the threshold itself.

**Two corrections to my own reasoning, both caught by measuring.** First, the f32 `&&` → `||`
mutant looked like the f32 peer of the defect above and is **not one**: the Stirling difference is
accurate far below its own threshold — `9.9e-7` relative with `a + n` as low as `2` — so forcing
it on is a shift of about eight `f32` ulps, not an error. Two hand calculations said otherwise and
both had an off-by-one in `lnΓ`, using `lnΓ(100)` where `lnΓ(101)` was needed. Second, a
continuity test compared `a = 299` against `a = 301` and failed at `n = 250` by `1.2` — which is
the function's own slope, `ψ(a+n) − ψ(a) ≈ 0.6`, not a step at the join. A continuity test has to
step by an **ulp**.

The seven left: four in the non-finite guard's arithmetic and the f32 `&&`, all of which have
descriptions colliding with real mutants elsewhere in the same function, plus two that want the
instrument this file prescribes and this work has not built — **a high-precision reference**. The Stirling branch runs only for *non-integer* `n` at large `a`, where the direct
sum does not apply and the `lnΓ` difference has already lost its digits — which is exactly why
the branch exists. There is no elementary reference in that regime, so its correction terms
(`1/(12(a+n)) − 1/(12a)`) are unreachable by any identity. That is a `bin/references/` job.

The rest are the usual regime comparisons, including `n > 0.0` relaxed to `>=` where `n == 0`
has already returned — unreachable, and awaiting the differential-harness treatment rather than
an argument.

#### Airy: 1,060 mutants this file had never mentioned, and the Wronskian that reaches them

A coverage sweep of the public surface against the generated suites found the largest blind spot
in the crate: **164 required methods on `Real`, 28 named in any generated property.** Of the 136
uncovered, 35 are constant-like and not property-shaped, which leaves about a hundred functions
with no generated law — and the Airy and Bessel families, 22 methods between them, had *none*.

`kernels/airy*` holds **1,060 mutants**, about 16% of euclid-scalar's whole population, and this
file did not mention Airy once before this entry.

**The instrument that reaches them is the Wronskian**, and it is a different class of check from
everything else here. Its value is a closed form — `1/π`, `1/x`, `−2/(πx)` — so it needs no
reference value, and *one* identity constrains four separately-implemented functions
simultaneously: `Ai`, `Ai′`, `Bi`, `Bi′` are four code paths with their own series, asymptotics
and crossovers, and the identity is false unless all four agree with each other and with the
equation they solve.

The scaled variants satisfy the **same** identity, which is what makes the whole axis reachable.
They are defined with reciprocal factors — `Aiᵉ = e^{ζ}Ai` against `Biᵉ = e^{−ζ}Bi`,
`I₀ᵉ = e^{−|x|}I₀` against `K₀ᵉ = e^{x}K₀` — so every product pairs one of each and the factor
cancels exactly. `Ai` underflows past `x ≈ 104` and `Bi` overflows there; the scaled pair are
ordinary numbers at `1e12`. An unscaled `I`/`K` sweep skipped 1,716 of 2,001 points above
`x = 100`; the scaled one is clean from `1e-8` to `1e15`.

**The whole surface has now been measured, and the Wronskians carried it.** `airy.rs` and
`airy/series.rs` together: **915 mutants, 906 caught, 0 timeouts, 0 unviable, 9 survivors.** With
`march.rs` that is **1,060 mutants and 12 survivors — a 98.9% kill rate on a file tree that had
no generated property at all a day earlier.**

That is the strongest single result in this file, and it is worth being precise about the cause:
these are not tests that were already there. Airy had *zero* generated coverage and no entry in
this document. Four Wronskian properties and two scaling ones took 1,060 mutants to twelve,
because the identities constrain all eight functions *against each other* — there is nowhere for
a wrong branch, a wrong coefficient or a wrong crossover to hide when `Ai·Bi′ − Ai′·Bi` has to
come out to `1/π` at every argument on the line.

**Triaged, and one of the nine was a real defect.** The differential treatment the paragraph
below asked for has now been applied, and the Airy tree stands at **909 mutants after exclusions,
907 caught, 2 survivors**.

The defect is a **signed zero**, and it is the third of this shape in this file. Above
`LEADING_MIN = 1e10` the kernel returns the limits directly rather than multiplying a finite
scaled value by an infinite exponential, and that tuple is `(0.0, -0.0, +∞, +∞)`. Deleting the
minus — making `Ai′` a *positive* zero — survived all 1,060 mutants, because no comparison can
see it: `-0.0 == 0.0`. The sign is not decoration. `Ai` is positive and decreasing on the
positive axis, so `Ai′ < 0` there and the limit is approached from below; a caller taking its
sign to choose a direction gets the wrong answer. Pinned now by
`the_airy_derivative_reaches_zero_from_below`, alongside the companion claims that `Ai` reaches
`+0.0`, that `Bi` and `Bi′` both diverge upward, and that `Ai′` is negative at moderate arguments
too — so the limit is consistent rather than arbitrary.

Six more are excluded on a committed harness in `kernels/airy/tests.rs`, and one of them has a
better explanation than the one first reached for. `Biᵉ` is `√(x/3)·(I_{−1/3}(ζ) + I_{1/3}(ζ))`,
and dropping either minus makes it `2·I_{1/3}(ζ)`. The first diagnosis was asymptotic — the
difference `I_{−ν} − I_ν = (2/π)sin(νπ)K_ν` decays like `e^{−2z}`, and the branch runs only where
`ζ ≥ 18`, so `e^{−36} ≈ 2e-16`. True, but the real reason is **structural**:
`bessel::asymptotic::i_scaled` reaches its order only through `4ν²`, so `i_scaled(−ν, x)` and
`i_scaled(ν, x)` are the same computation bit for bit. The written expression documents the
mathematics while the implementation collapses it, which is correct here and worth a reader
knowing rather than rediscovering from a surviving mutant.

**Both remaining pairs have since been settled, and the Airy tree is down to three visible
survivors — all three proven, and left visible on purpose.**

`fg`'s convergence break is now measured rather than argued.
`the_series_break_cannot_change_the_series` transcribes the loop with the break under a selector
and compares all three variants bitwise across 4,006 arguments spanning `±SERIES_MAX`, both zeros
and the boundary. The `==` variant is the one that matters: it disables the break entirely, so the
loop runs to `MAX_TERMS` and the comparison measures exactly what the discarded terms were worth.
Zero bits, everywhere. Both are now excluded.

`march`'s three needed the reachability step the previous pass correctly refused to skip. The
truncation bound in `ORDER`'s docstring, `|h|^{N+1}/(N+1)!`, ignores the coefficient growth in
`c[n+2] = (x₀c[n] + c[n−1])/((n+2)(n+1))` — and a bound that ignores the thing that could break it
is not a bound. The missing fact is the *domain*: `march::airy` marches from `−SERIES_MAX = −2`
outward to at most `MARCH_MAX = 9` in steps of at most `MAX_STEP = 0.5`, so `|x₀| ≤ 9` and
`|h| ≤ 0.5` and nothing larger is ever reached. At order 24 the term is then about
`9¹²/24! · 0.5²⁴ ≈ 3e-20`, four orders below an ulp. The growth is real; the factorial still wins
at the reachable arguments. Measured over that domain —
`the_marchs_top_taylor_terms_are_below_resolution_where_it_runs` — the dropped terms move nothing
by more than half an ulp.

**And yet they stay in the report, because no `exclude_re` pattern can isolate them.** The three
are `+ with *` at `ORDER + 2`, `+ with *` at `ORDER + 1`, and `+ with -` at the same place — and
`step` contains several other `+` sites, including `x0 * c[n] + previous`, where the identical
`+ with *` mutation is a real defect that the suite *does* catch. The description names the
operator and the function and cannot distinguish the line, so excluding the pattern would hide a
genuine finding the moment that line changed. Same situation as `ldexp`'s two `||`s and the f32
`ndtri` narrowing, and the same resolution: three documented survivors are better than one hidden
defect.

The nine as first measured, all in the quadrature and the series' crossover comparisons:

| Where | Count |
|---|---|
| `quad` and `quad_scaled` — the trapezoidal integrator's loop bounds and a deleted negation | 4 |
| `bi_pair_scaled` — two deleted negations | 2 |
| `positive_scaled`, and `fg`'s `<` relaxed to `<=` and narrowed to `==` | 3 |

Not yet diagnosed per site. The shape is familiar — loop bounds and regime comparisons, the
family this file has repeatedly found to be either sub-ulp or unreachable — but *familiar is not
measured*, and the `ndtri` and `erfc_inv` findings in the normal family are what that warning is
based on: two mutations with exactly this shape turned out to be real defects after being filed
as equivalent by argument. These nine want the differential-harness treatment before any of them
is excluded.

The first measurement, on `airy/march.rs` alone — the Taylor-march integrator that produces all
four: **145 mutants, 140 caught, 2 timeouts, 3 survivors.** The three are worth naming because
none is a missing assertion:

| Survivor | Reading |
|---|---|
| `ORDER + 2` → `* 2` in the coefficient array's length | slack. The array needs 26 entries and gets 48; the extra are never written or read. Provably equivalent. |
| `ORDER + 1` → `* 1` in the Horner loop bound | drops the top Taylor coefficient `c[25]` |
| `ORDER + 1` → `- 1` in the same bound | drops `c[24]` and `c[25]` |

The second and third look like [cause 7](#7-dead-precision--arithmetic-the-format-cannot-represent) —
the truncation at `ORDER = 24` with `|h| ≤ ½` is below `1e-40`, twenty-four orders under `ulp(1)`,
so the last terms cannot matter. **But that bound ignores coefficient growth**, and the
recurrence `c[n+2] = (x₀c[n] + c[n−1])/((n+2)(n+1))` grows with `x₀`: at large `|x₀|` the terms
rise before they decay and the series genuinely needs its length. So these are dead precision *at
the arguments the march is reached with*, which is not the same claim as dead in general.
Distinguishing them needs the `x₀` range the march actually sees — the cheap reachability
experiment [carry-forward 3](#what-to-carry-forward) prescribes — and that has not been run. Not
excluded.

#### `integrals.rs` and `normal.rs`: the founding example, found again

The last never-diagnosed row, measured: **505 mutants** across `kernels/integrals.rs` (291),
`kernels/normal.rs` (101), `kernels/f32/normal.rs` (101) and `kernels/f32/integrals.rs` (12) —
**452 caught, 53 survivors**, now **49** after the compensated core below (`integrals.rs` alone:
291 mutants, 20 survivors → **16**, confirmed by re-run). Where they sat:

| Function | Survivors | Reading |
|---|---|---|
| `two_sum` + `compensated_series` | 11 | the `compensated_sum` shape — see below |
| `tail_start` (both precisions) | 11 | cause 8, a Newton seed; one is a *constant* |
| `log_ndtr` (both) | 7 | the branch crossovers at `x = 0` and `x = -1` |
| `ndtri` + `ndtri_lower` (both) | 12 | the domain guard, and the seed selection at `p >= 1e-15` |
| `exp_int_ei`, `sin_int`, `trig_integrals`, `e1_continued_fraction` | 9 | regime crossovers |
| `erfc_inv` (both) | 2 | the domain guard |

**Eleven of them are in the compensated summation core, which is where this whole exercise
started.** `.cargo/mutants.toml` exists because `euclid-stability`'s `compensated_sum` was
executed by three tests and pinned by none. `integrals.rs` has its own local `two_sum` — "local
so this module does not depend on the trait surface" — and *all three* of its mutations survived:
`a + b` to `a - b`, to `a * b`, and the residual's `+` to `*`.

The reason is worth internalising, because it generalises to every compensated algorithm:
**`two_sum` is self-correcting in its total.** Whatever the primary sum becomes, the residual
absorbs the difference, and `s + e` still reconstructs `a + b`:

| | `s` | `e` | `s + e` |
|---|---|---|---|
| correct | `0.30000000000000004` | `-2.78e-17` | `0.30000000000000004` |
| `a + b` → `a - b` | `-0.1` | `0.4` | `0.30000000000000004` |
| `a + b` → `a * b` | `0.020000000000000004` | `0.28` | `0.30000000000000004` |

Every caller adds the two halves straight back together, so a test that checks a *sum* cannot
see any of this. What the mutations destroy is the **decomposition**, and that is the entire
product: `compensated_series` accumulates `total` and `correction` separately across terms, so a
residual carrying `0.4` instead of `-2.78e-17` makes the compensation meaningless and the error
`O(n·ε·Σ|tₖ|)` again. `correction += e` was likewise replaceable with `correction *= e`, which
zeroes the compensation outright since `correction` starts at zero.

Closed by asserting the two halves rather than their sum. `two_sum`'s identity needs no reference
values — where `a + b` is representable the residual is exactly zero, and where it needs a 54th
bit the residual is exactly the part that did not fit — and `compensated_series` is pinned by the
standard witness: one unit term plus a thousand at `1e-17`, each below half an ulp of the running
total, so naive summation returns exactly `1.0` while the compensated answer is `1 + 1e-14`. The
exact sum is known by construction, so again no reference is needed.

One detail that cost a first draft: the recovered amount can only be read back to **one ulp of
`1.0`**. `1 + 1e-14` is 45 ulps above `1.0`, so `compensated - naive` is `9.992e-15` for an exact
`1e-14`, and an assertion tighter than `f64::EPSILON` there is asserting more than the format
holds. Still a hundred-million-fold margin over the signal, which is compensation against none.

**Also worth noting: this `two_sum` is a second copy.** `tests::compensated::primitives` already
pins a `two_sum` in the crate's compensated module, with three tests including
`two_sum_is_exact`. The duplicate in `integrals.rs` inherited none of it — cause 1, in the form
where the untested twin is a *deliberate* local copy rather than a generated default.

That closes four of the eleven. The seven left in `compensated_series` are the `magnitude`
accumulator (2) and the stopping rule `term.abs() < magnitude * 1e-20 && k > 2` (5), and both
look like the defensive-guard family rather than gaps: `magnitude` gates only the *early exit*,
so `*=` leaves it at zero and the loop simply runs until the generator returns `None` — the same
total, more slowly — and the `k > 2` boundary moves the exit by one or two terms whose magnitude
is already below `1e-20` of the running sum. Plausible, not established; neither is excluded.

The remaining 49 are not yet closed, and two readings are worth not losing:

- **`tail_start -> -1.0` survives at f32 and is caught at f64.** A Newton seed replaced by a
  constant, and six steps on the log-CDF still satisfy the f32 suite at `p = 1e-30` — because
  that suite's tolerance is `1e-4` relative on a value of `-69`, which is `7e-3` absolute. The
  f64 peer's deep-tail test does notice. This is cause 8 with a number attached, and it frames
  the question correctly: the seed is unpinned *to the tolerance currently asserted*, which is
  not the same claim as unpinnable. **Now settled — see below.**

#### The normal family, 33 to 1: where a property failed, a unit test did not, and the triage was wrong twice

The cleanest experiment in this file, because both halves were measured.

**A generated property killed nothing.** `tests/properties/identities/normal.rs` was written
first: ten properties over `ndtr`, `log_ndtr`, `ndtri` and `erfc_inv`, including a round trip
sweeping `p` from `1e-300` in log space at a `1e-14` relative bound, a coverage floor of 15% on
the deep-tail seed branch specifically, and the left-tail asymptotic as an independent reference.
Re-measured after: **33 survivors before, 33 after, byte-for-byte the same set.**

That is the empirical answer to the question the row above left open, and it is worth stating
plainly: for a self-correcting iteration, **widening the domain does not reach the seed either.**
Not a tolerance problem and not a coverage problem — six Newton steps on `ln Φ(x) = ln p` converge
from any guess in the right region, so no assertion on the *output* can constrain the input. The
properties are still worth having (they pin ten laws nothing else states, and they found the
signed zero in `log_ndtr(+∞)`), but they are not the instrument for this cluster.

**Two unit tests killed twelve.** The seed is a private function, so the kernel's own test module
can call it directly — which is the whole trick, and it is available precisely because
`tail_start` is *named*. The law needs no reference: a seed is good when `ln Φ` of it is close to
`ln p`, which is the equation being solved. Measured over 3,000 points on `[1e-300, 1e-15]`: worst
relative error `1.07e-3` in `ln p`, worst distance from the converged root `4.6e-3` — and the
docstring's own claim, "good to a few digits — ample for Newton to finish from", had never been
checked. The mutations miss by whole units: `/` for `*` puts the seed at `-2.34` where the root is
`-7.94`. Bounds an order looser than the measurement kill all twelve.

A thirteenth was a plain gap: f32 `log_ndtr`'s `(-0.5 * erfc(x/√2)).ln_1p()` had its `*`
replaceable with `+` because **nothing in `f32` evaluated the upper tail** — the f32 suite tested
`log_ndtr(-20)` and nothing above zero, while the f64 peer is covered. Inherited-nothing again.
The mutated form returns `-0.69` where the answer is `-6.9e-4`.

**Then a differential harness found two defects the triage had called equivalent.** This is the
part worth carrying forward, because the triage was mine and it was careful, and it was wrong
twice.

The remaining survivors were all guard and crossover comparisons, each with a reachability
argument behind it. `.cargo/mutants.toml` does not accept an argument, so the search went into
`the_excluded_normal_mutants_change_no_observable_value` — in Rust rather than
`bin/equivalence.py`, because these bodies call `erfc`, `erfcx` and `erf_inv` and transcribing
those into Python is transcribing the thing under test. It sweeps 2,350 arguments: both zeros, the
subnormals, forty ulps either side of all three branch boundaries, a 320-decade log sweep in both
signs, and the central region linearly.

It disagreed with the reasoning twice:

- **`ndtri`'s `p > 0.5` narrowed to `p == 0.5` is real.** The argument had been that
  `ndtri_lower` solves correctly for `p > 0.5` too, so both routes agree. They do, until
  `p = 1 − ulp(1)`, where the direct solve returns `8.1824` and the answer is `8.2095` — wrong in
  the second decimal. Newton's residual is `ln Φ(x) − ln p`, and near `x = 8.2` both terms are
  about `−1e-16`, so the residual is a difference of two quantities that have each lost all but a
  couple of digits. The symmetric route evaluates the same root as `−ndtri_lower(1 − p)`, where
  `1 − p` is `1.1e-16` and the lower tail is perfectly conditioned. Nothing had evaluated the
  quantile that close to one: the fixed tests stop at `0.99` and the generated property draws
  `p ≤ 0.5`.
- **`erfc_inv`'s `||` weakened to `&&` is real, and returns `+∞`.** The argument had been that an
  out-of-range argument reaches the same `NaN` through `ln` of a negative. It does for `-0.1` and
  `2.5`, which is what the existing domain tests use. It does not for `-5e-324`: `0.5 · y`
  underflows to `−0.0`, `ndtri(−0.0)` takes the `p == 0.0` branch and returns `−∞`, and the
  closing negation turns that into a confident `+∞` for an argument outside the domain. Only an
  argument whose *half underflows to a signed zero* reaches the branch that invents an answer.

Both are now killed, in both precisions. The lesson is
[carry-forward 1](#what-to-carry-forward) with a sharper edge than it had: computing the answer
beat reasoning about the source, and the reasoning that lost was not sloppy — it was a
reachability argument of exactly the kind this file accepts elsewhere. **A differential search is
cheap enough that an exclusion should never rest on an argument alone.**

The same harness also corrected the *form* of the equivalence it did confirm. `log_ndtr`'s
crossovers are equivalent **to within two ulps, not bitwise**: at `x = -0.9999999999999996` the
direct-log branch and the `erfcx` one differ by one ulp. Both are correct there and neither is the
reference, which is what makes the switch a free choice — and a first draft of the test asserted
bit equality and failed on that argument.

**Where the family landed: 202 mutants, 1 visible survivor.**

| Action | Count |
|---|---|
| `tail_start` seeds, killed by testing the seed as a function | 12 |
| `central_start` seeds, killed after extracting it into a named function | 4 |
| f32 `log_ndtr`'s upper tail, a plain gap | 1 |
| the two defects above, killed in both precisions | 4 |
| seed *selection*, excluded — guaranteed by a seed-independence test | 2 |
| crossovers and redundant guards, excluded on the differential search | 11 |
| left visible on purpose | **1** |

That last one is `replace > with == in ndtri` at **f32**, and it is the `ldexp` situation again:
the identical mutation is *real* at f64 (above) and *equivalent* at f32, because f32 cannot reach
a `1 − p` small enough for the residual to collapse — the two routes agree to `9e-8` relative,
under one ulp. One `exclude_re` pattern cannot distinguish them, and excluding it would hide the
f64 defect the moment someone touched that branch. So it stays in the report with the reason
written down.

**And a contract gap, recorded rather than fixed.** `erfc_inv(5e-324)` returns `+∞` where the true
abscissa is about `27.3`. The limit is the halving, not the solver — `ndtri` itself is good to
`1e-320` — so the function is narrower than its docstring's claim to stay "accurate as far into
the tail as `ndtri` does". Left as it behaves because the fix is a decision: scaling by `0.5`
exactly is what makes `erfc⁻¹(y) = −Φ⁻¹(y/2)/√2` hold, and avoiding the underflow means
reformulating in the log domain or accepting a different rounding everywhere else.

#### `integrals.rs` and `normal.rs`: the founding example, found again

The last never-diagnosed row, measured: **505 mutants** across `kernels/integrals.rs` (291),
`kernels/normal.rs` (101), `kernels/f32/normal.rs` (101) and `kernels/f32/integrals.rs` (12) —
**452 caught, 53 survivors**, now **49** after the compensated core below (`integrals.rs` alone:
291 mutants, 20 survivors → **16**, confirmed by re-run). Where they sat:

| Function | Survivors | Reading |
|---|---|---|
| `two_sum` + `compensated_series` | 11 | the `compensated_sum` shape — see below |
| `tail_start` (both precisions) | 11 | cause 8, a Newton seed; one is a *constant* |
| `log_ndtr` (both) | 7 | the branch crossovers at `x = 0` and `x = -1` |
| `ndtri` + `ndtri_lower` (both) | 12 | the domain guard, and the seed selection at `p >= 1e-15` |
| `exp_int_ei`, `sin_int`, `trig_integrals`, `e1_continued_fraction` | 9 | regime crossovers |
| `erfc_inv` (both) | 2 | the domain guard |

**Eleven of them are in the compensated summation core, which is where this whole exercise
started.** `.cargo/mutants.toml` exists because `euclid-stability`'s `compensated_sum` was
executed by three tests and pinned by none. `integrals.rs` has its own local `two_sum` — "local
so this module does not depend on the trait surface" — and *all three* of its mutations survived:
`a + b` to `a - b`, to `a * b`, and the residual's `+` to `*`.

The reason is worth internalising, because it generalises to every compensated algorithm:
**`two_sum` is self-correcting in its total.** Whatever the primary sum becomes, the residual
absorbs the difference, and `s + e` still reconstructs `a + b`:

| | `s` | `e` | `s + e` |
|---|---|---|---|
| correct | `0.30000000000000004` | `-2.78e-17` | `0.30000000000000004` |
| `a + b` → `a - b` | `-0.1` | `0.4` | `0.30000000000000004` |
| `a + b` → `a * b` | `0.020000000000000004` | `0.28` | `0.30000000000000004` |

Every caller adds the two halves straight back together, so a test that checks a *sum* cannot
see any of this. What the mutations destroy is the **decomposition**, and that is the entire
product: `compensated_series` accumulates `total` and `correction` separately across terms, so a
residual carrying `0.4` instead of `-2.78e-17` makes the compensation meaningless and the error
`O(n·ε·Σ|tₖ|)` again. `correction += e` was likewise replaceable with `correction *= e`, which
zeroes the compensation outright since `correction` starts at zero.

Closed by asserting the two halves rather than their sum. `two_sum`'s identity needs no reference
values — where `a + b` is representable the residual is exactly zero, and where it needs a 54th
bit the residual is exactly the part that did not fit — and `compensated_series` is pinned by the
standard witness: one unit term plus a thousand at `1e-17`, each below half an ulp of the running
total, so naive summation returns exactly `1.0` while the compensated answer is `1 + 1e-14`. The
exact sum is known by construction, so again no reference is needed.

One detail that cost a first draft: the recovered amount can only be read back to **one ulp of
`1.0`**. `1 + 1e-14` is 45 ulps above `1.0`, so `compensated - naive` is `9.992e-15` for an exact
`1e-14`, and an assertion tighter than `f64::EPSILON` there is asserting more than the format
holds. Still a hundred-million-fold margin over the signal, which is compensation against none.

**Also worth noting: this `two_sum` is a second copy.** `tests::compensated::primitives` already
pins a `two_sum` in the crate's compensated module, with three tests including
`two_sum_is_exact`. The duplicate in `integrals.rs` inherited none of it — cause 1, in the form
where the untested twin is a *deliberate* local copy rather than a generated default.

That closes four of the eleven. The seven left in `compensated_series` are the `magnitude`
accumulator (2) and the stopping rule `term.abs() < magnitude * 1e-20 && k > 2` (5), and both
look like the defensive-guard family rather than gaps: `magnitude` gates only the *early exit*,
so `*=` leaves it at zero and the loop simply runs until the generator returns `None` — the same
total, more slowly — and the `k > 2` boundary moves the exit by one or two terms whose magnitude
is already below `1e-20` of the running sum. Plausible, not established; neither is excluded.

The remaining 49 are not yet closed, and two readings are worth not losing:

- **`tail_start -> -1.0` survives at f32 and is caught at f64.** A Newton seed replaced by a
  constant, and six steps on the log-CDF still satisfy the f32 suite at `p = 1e-30` — because
  that suite's tolerance is `1e-4` relative on a value of `-69`, which is `7e-3` absolute. The
  f64 peer's deep-tail test does notice. This is cause 8 with a number attached, and it frames
  the question correctly: the seed is unpinned *to the tolerance currently asserted*, which is
  not the same claim as unpinnable. **Now settled — see below.**

#### The normal family, 33 to 1: where a property failed, a unit test did not, and the triage was wrong twice

The cleanest experiment in this file, because both halves were measured.

**A generated property killed nothing.** `tests/properties/identities/normal.rs` was written
first: ten properties over `ndtr`, `log_ndtr`, `ndtri` and `erfc_inv`, including a round trip
sweeping `p` from `1e-300` in log space at a `1e-14` relative bound, a coverage floor of 15% on
the deep-tail seed branch specifically, and the left-tail asymptotic as an independent reference.
Re-measured after: **33 survivors before, 33 after, byte-for-byte the same set.**

That is the empirical answer to the question the row above left open, and it is worth stating
plainly: for a self-correcting iteration, **widening the domain does not reach the seed either.**
Not a tolerance problem and not a coverage problem — six Newton steps on `ln Φ(x) = ln p` converge
from any guess in the right region, so no assertion on the *output* can constrain the input. The
properties are still worth having (they pin ten laws nothing else states, and they found the
signed zero in `log_ndtr(+∞)`), but they are not the instrument for this cluster.

**Two unit tests killed twelve.** The seed is a private function, so the kernel's own test module
can call it directly — which is the whole trick, and it is available precisely because
`tail_start` is *named*. The law needs no reference: a seed is good when `ln Φ` of it is close to
`ln p`, which is the equation being solved. Measured over 3,000 points on `[1e-300, 1e-15]`: worst
relative error `1.07e-3` in `ln p`, worst distance from the converged root `4.6e-3` — and the
docstring's own claim, "good to a few digits — ample for Newton to finish from", had never been
checked. The mutations miss by whole units: `/` for `*` puts the seed at `-2.34` where the root is
`-7.94`. Bounds an order looser than the measurement kill all twelve.

A thirteenth was a plain gap: f32 `log_ndtr`'s `(-0.5 * erfc(x/√2)).ln_1p()` had its `*`
replaceable with `+` because **nothing in `f32` evaluated the upper tail** — the f32 suite tested
`log_ndtr(-20)` and nothing above zero, while the f64 peer is covered. Inherited-nothing again.
The mutated form returns `-0.69` where the answer is `-6.9e-4`.

The remaining 20 are triaged and none is a missing assertion:

| Where | Count | Why |
|---|---|---|
| `log_ndtr`'s branch comparisons, both precisions | 6 | the crossovers at `x = 0` and `x = -1`; at `x = 0` both branches return `ln(0.5)` by different routes |
| `ndtri`/`erfc_inv`'s `\|\|` guards, both precisions | 4 | redundant with NaN propagating through `ln` of a negative |
| `ndtri`'s `p > 0.5` symmetry branch, both precisions | 4 | `p == 0.5` returned above it, and `ndtri_lower` solves correctly for `p > 0.5` anyway, so both routes agree |
| `ndtri_lower`'s **central** seed | 6 | cause 8 again — but this seed is written inline rather than named, so no test can call it |

That last row is the actionable one, and it is a conclusion about structure rather than about
tests: **the central seed is unpinnable for exactly as long as it is an expression instead of a
function.** Extracting it the way `tail_start` already is would make the same two-line test apply.
That is a change to the kernel for testability, which is why it is recorded here rather than done
in passing.
- **The domain guards are redundant rather than untested**, which was not the first guess. The
  f64 suite *does* assert `ndtri(1.5)`, `ndtri(-0.5)`, `erfc_inv(2.5)` and `erfc_inv(-0.1)` are
  NaN, and the `||`-to-`&&` mutants still survive: with the guard weakened, an out-of-range `p`
  falls through to `ln(p)` of a negative, and NaN propagates to the same answer by a different
  route. Equivalent, by a reachability argument — **not** brute-force confirmed, so left visible
  rather than excluded, the same standing as the `beta_cf` Lentz guards. Worth recording that the
  carry-forward rule below predicted a test gap here and was wrong; the check is cheap and the
  prediction was not evidence.

#### The integrals' sixteen, and the breadth check that an exclusion needs

`integrals.rs` closed from sixteen survivors to **three**, and the interesting part is the three.

Thirteen are excluded on `kernels/integrals/tests.rs`, which transcribes `compensated_series`
with its mutations under a selector and compares bitwise against the real function. The claims
are the familiar ones — `magnitude` is never summed into the result, so moving the stopping rule
changes iteration count and nothing else; `exp_int_ei`'s `x < 0.0` cannot see a zero because
`x == 0.0` returned `−∞` above it; `sin_int`'s sign test sits inside `if ax.is_infinite()`; the
two regime crossovers are measured to agree at `1e-13` at the exact argument that moves between
them. `term > prev`, the divergence guard, differs only at `k == x` — an integer `x ≥ 40` — where
the extra term is `40!/40^40 ≈ 8e-17` against a total of about one.

**One draft failure is worth recording, because it is the same mistake for the fourth time.** The
harness first swept the trigonometric generators to `x = 39.9` and failed on `Ci` at `x = 20` by
`1.3e-15` — correctly. `TRIG_SERIES_MAX` is `4`, and past it those terms grow to `20²⁰/(20·20!)`
before they shrink, so the early exit genuinely matters there. It is simply a range the kernel
never evaluates that series on. Together with the `ulp` generator, the rising factorial's integer
`n` and `ln_rising_factorial`'s non-finite guard, the rule is now unmistakable: **the question for
a branch test is not "does the law hold" but "on which arguments does this code run".**

#### Checking a pattern's breadth before excluding, which this file should have said earlier

The three that remain are proven equivalent by the same tests and **deliberately not excluded**,
and finding that out took a step this file has never written down:

```sh
cargo mutants --list -f <file> | grep 'in <fn>$' | sort | uniq -c
```

That counts how many *instances* a description matches. An exclusion is only safe when every
instance of its pattern is a survivor, because `exclude_re` matches the description — the operator
and the function — and never the line.

| Pattern | Instances | Survivors | Safe |
|---|---|---|---|
| `replace < with <= in exp_int_ei` | 3 | 3 | yes |
| `replace < with == in exp_int_ei` | 3 | 1 | **no** |
| `replace += with -= in compensated_series` | 2 | 1 | **no** |
| `replace += with *= in compensated_series` | 2 | 1 | **no** |

The two `compensated_series` collisions are the sharpest case in this document. One instance of
each is `magnitude`, which is unobservable; the other is `correction` — the compensation itself,
where `correction *= e` zeroes it outright. That is the founding defect of this whole exercise,
now pinned by `compensated_series_recovers_terms_naive_summation_drops_entirely`. Excluding the
pattern to tidy away the `magnitude` survivor would have re-hidden it.

So three documented survivors stay. With `march`'s three, `ldexp`'s `||`, the f32 `ndtri`
narrowing and `polygamma`'s pair, that is the standing population of *proven-equivalent but
unexcludable* mutants — and the honest floor this report has, as against the kind that trains a
reader to skim.

#### The ten equivalents, and the instrument that establishes them

`.cargo/mutants.toml` refuses an exclusion that is only argued for. The argument for these ten is
short enough to be suspicious — each is a comparison whose boundary an earlier `return` has
already removed — so they are settled by **`bin/equivalence.py`, now committed**:

| Mutation | Inputs searched | Verdict |
|---|---|---|
| `frexp`: the splice `\|` → `^` | every input, by masks | equivalent, **exhaustively** |
| `next_after`: all three `>` → `>=` | 208,849 f64 / 4,165,681 f32 pairs | equivalent |
| `ulp`: `x < 0.0` → `<=` | 457 f64 / 2,041 f32 values | equivalent |
| `ldexp`: `* → /`, `> → >=`, `< → <=` | 2,027,252 f64 / 1,706,276 f32 pairs | equivalent |

The `frexp` splice is the unusual one: `(bits & !EXP_BITS) | (HALF_BIAS << SHIFT)` ORs two
operands whose overlap is a *constant*, `(!EXP_BITS) & (HALF_BIAS << SHIFT)`, and it evaluates to
`0x0` at both precisions. A zero overlap settles all 2⁶⁴ and 2³² inputs at once, so unlike every
other search in this file there is no residual doubt about an untried value. The rest are sampled
and say so.

Two things about the script are the point, not the detail:

- **It asserts the real mutations differ, not only that the equivalent ones agree.** The models
  are transcriptions of the Rust, and a transcription can be wrong in the same direction as the
  thing it checks; a harness that can only report "no difference" cannot tell a proof from a
  broken search. So `ldexp`'s second `||` (904 f64 / 4,072 f32 differing pairs), `ulp`'s
  `< → ==` (55 / 254), `frexp`'s `||` (5 / 5) and `next_after`'s NaN guard (912 / 4,080) are all
  required to be *detected*, and the exit code is non-zero if any of them goes silent. This is
  the self-validation rule `bin/references/incomplete.py` already follows, applied to an
  equivalence search.
- **It covers both precisions, because the exclusions do.** `exclude_re` matches a mutation's
  description, which names the function and not the file, so `replace > with >= in ldexp`
  excludes the f32 mutant as well as the f64 one. The comment in `.cargo/mutants.toml` previously
  cited an f64-only search while excluding both — a smaller version of the mistake the f32 row of
  the aggregate table records, and corrected the same way: by measuring.

#### What is left

Against the aggregate's 584, in descending order:

| Open | Count | Needs |
|---|---|---|
| `kernels/f32/quantile.rs` | 73 (was 57) | nothing: cause 8, confirmed by experiment — the rise *is* the fix landing |
| `kernels/quantile.rs` | 56 | nothing: equivalent by design, [cause 8](#8-a-self-correcting-iteration-whose-inputs-cannot-reach-its-output) |
| `kernels/transcendental.rs` (Lambert W) | **36** (was 43) | [measured](#lambert-w-and-zeta-the-best-law-in-the-crate-and-a-seed-that-defeats-three-instruments): 16 seed, 12 Halley step, 8 zeta — the seed wants its regime narrowed, not a test |
| `kernels/f32/gamma.rs` + `gamma.rs` + the polygamma pair | **76** (was 90) | [measured and triaged](#the-gamma-family-eighteen-laws-and-the-lesson-that-a-true-law-can-test-nothing): a negative-count defect found; two of `ln_rising_factorial`'s remaining seven want a high-precision reference |
| `kernels/f32/polygamma.rs` + `kernels/polygamma.rs` | 24 (was 52) | Euler–Maclaurin coefficient signs, and one f64 `exprel` crossover needing a finer reference |
| `kernels/bessel/modified_first.rs` + f32 peer | 39 (was 47) | convergence guards on `ratio < sum * TERM_EPS`; the sign work is done |
| `kernels/gamma.rs` | 23 | per-site triage |
| `kernels/repr.rs` + `f32/repr.rs` | **2** (was 17) | nothing: both are the `ldexp` guard left visible on purpose, [above](#the-representation-pair-17-survivors-and-a-guard-tested-from-only-one-side) |
| `kernels/airy*` | **3** of 1,052, all proven | [fully triaged](#airy-1060-mutants-this-file-had-never-mentioned-and-the-wronskian-that-reaches-them) — one defect found, eight excluded on harnesses; `march`'s three are proven equivalent but unexcludable, since the pattern would hide a real mutant on the same line |
| `kernels/normal.rs` + `f32/normal.rs` | **1** (was 33) | [closed](#the-normal-family-33-to-1-where-a-property-failed-a-unit-test-did-not-and-the-triage-was-wrong-twice) — two defects found, the rest killed or excluded on a committed differential search; the one left is visible on purpose |
| `kernels/integrals.rs` + `f32/integrals.rs` | **3** (was 20) | [fully triaged](#integralsrs-and-normalrs-the-founding-example-found-again) — thirteen excluded on a transcribed harness; the three left are pattern collisions with real mutants on neighbouring lines |
| `conformance.rs`, the `real_laws` family | **0** (was 7) | [closed](#the-real_laws-seven-a-generated-carrier-that-can-be-wrong-in-one-place) — a generated `Faulty<WHICH>` carrier, 28 of 35 caught and 7 unviable |

The f32 tree is where the work is: **234 of 584**, and the one cluster opened so far gave up 36
mutants to two measured bounds. `f32/polygamma.rs` and `f32/bessel` are the next two to look at,
and the first question for each is the cheap one — does the suite reach the branch at all.

**The `real_laws` seven all need the same thing, and it is expensive.** Every one of them —
`real_laws`, `real_laws_for`, `real_laws_generated`, both mutations of the tolerance bound, and
the two `delete match arm` mutants — is only reachable by a carrier that implements `Real` and is
wrong in one place. There is no non-primitive `Real` implementation anywhere in the workspace, so
there is nothing to wrap: the tower is **153 required methods on `Real`, plus 5 on
`NormedDivisionAlgebra`, 8 on `Compare`, 2 on `Element`, and five operator impls** — 168 methods
before the carrier can misbehave in one corner. That is `euclid-algebra`'s `Faulty<WHICH>`
problem an order of magnitude larger, and the delegation should be *generated* from
`traits/real.rs` rather than hand-written, with the generator committed.

Two of the seven are delicate beyond that, and worth flagging before someone attempts them. The
`delete match arm "floor"` and `"ceil"` mutants make `floor` fall through to `_ => trunc`, so
killing them needs a carrier that **satisfies every law** — otherwise the rejection test passes
with and without the mutant and nothing is learned — while still making `trunc(floor(x)) ≠
floor(x)`. A broken `trunc` does not work: it fails the real suite too, so the test cannot tell
the two apart. The carrier has to be idempotent in all three roundings and still distinguish
their composition.

Priced honestly: the three whole-body mutants are the valuable ones, because `real_laws` is the
function a backend author runs to find out whether their carrier is sound, and a vacuous version
answers yes. The other four are refinements on an already-asserted suite.

#### The `real_laws` seven: a generated carrier that can be wrong in one place

**Closed.** `conformance.rs` is now **35 mutants, 28 caught, 7 unviable, none surviving**, from a
baseline of seven survivors that had resisted everything.

This was the highest-leverage entry in this file and the hardest to reach, for a reason that is
about exposure rather than score: `real_laws` is the function a backend author runs to find out
whether their carrier is sound, and a vacuous version answers "yes". The seven survived because
**nothing in the workspace implemented `Real` and was wrong**, so the laws had never been shown
to reject anything. A suite that cannot fail cannot answer the question it is asked.

`tests/vacuity.rs` already closed the `Boolean` and `Compare` tiers this way and said why it had
stopped: a broken `Real` has to carry **153 methods** before it can misbehave in one of them. So
the carrier is generated. `bin/carrier.py` reads the trait definitions and emits a
`Faulty<WHICH>` whose 160 methods delegate to `f64` except for method number `WHICH`, with
`--check` verifying the committed copy is current. Hand-writing it would have created a second
surface to keep in step with `Real`, and it would have rotted the first time a method was added.

**Three kinds of wrongness were needed, and discovering why is most of what this entry is worth.**

| Mode | Fault | Which mutants it reaches |
|---|---|---|
| coarse | `−(1.5v + 0.25)` | the whole-body ones: `real_laws`, `real_laws_for`, `real_laws_generated` |
| fine | `v·(1 + 1e-9)` | the **tolerance** ones |
| lawful-unusual | `trunc` rounds down to even | the two **`delete match arm`** ones |

- The coarse fault alone leaves the tolerance mutants alive. `real_laws` bounds each inexact law
  by `scale · tolerance`, and mutating that `*` to `+` makes the bound about `1` instead of
  `1e-12` — vacuous. A coarsely broken carrier is rejected under *either* bound, because it
  mostly produces `NaN` and `NaN ≤ anything` is false, so it cannot tell them apart. A fault of
  `1e-9` relative is three orders above the tolerance and nine below one, and is accepted exactly
  when the bound has been loosened.
- **An affine fault leaves `floor` idempotent.** The first attempt perturbed by `1.5v + 0.25` and
  the carrier passed: the broken `floor`'s image is `{0.25, 1.75}` and it maps that set to itself.
  Idempotence is the *only* law `real_laws` states about `floor`, so the perturbation sat in the
  law's kernel. The honest conclusion was not to loosen the test but to negate as well — and the
  finding stands on its own: **a `floor` that is wrong but idempotent satisfies every law this
  suite states.**
- The `delete match arm` pair needed the opposite direction entirely. Deleting the `"floor"` arm
  makes the check fall through to `_ => trunc`, so it becomes `trunc(floor(x)) == floor(x)` —
  true for every ordinary carrier, because `floor(x)` is an integer and `trunc` fixes integers.
  No *broken* carrier can see it. `Faulty<2000>` rounds `trunc` down to an even integer, which is
  idempotent, so every law holds and the real suite **accepts** it, while the mutated suite
  rejects it. Catching those two is an assertion about acceptance, which is the direction this
  file flagged as delicate.

**And a mistake worth recording, because it cost real coverage.** The first version of this work
wrote `tests/vacuity.rs` from scratch without checking whether the file existed. It did — 381
lines closing the `Boolean` and `Compare` tiers — and overwriting it silently removed **ten**
kills elsewhere in `conformance.rs`, which showed up as the file going from 21 caught to 13 while
the new tests all passed. The symptom was legible only because a before/after measurement was
taken; the per-slice verification this work has been doing is what caught it. Restored, and the
`Real` tier is now *added* to that file rather than replacing it.

#### Leads from the partial aggregate run

The full `euclid-scalar` pass was started and then invalidated — source changed under it
mid-run, which by this file's own rule makes it not evidence. It had tested 119 mutants before
being stopped, and three findings outside `conformance.rs` are worth not losing:

- **`impls/host/real.rs`** — `<f32 as HostReal>::is_finite`, `is_nan` and `is_sign_negative`
  replaceable with constants. The eager host surface was measured at 13 mutants with zero
  survivors *before* the merge; its f32 half is cause 3 again, a carrier beside the reference one.
- **`compensated.rs`** — `replace + with - in dot`. A compensated reduction whose sign can flip
  is exactly the `compensated_sum` shape that motivated `.cargo/mutants.toml` existing.
- **`interval.rs`** — `replace - with / in gap`.

None is established as a defect. They are where to start the clean run's triage.

#### The beta at large shape: a NaN for every `x`

Listed here as "47, accuracy cases above `BETA_EDGEWORTH_MIN`". `beta_reg` returned **NaN for
every `x`** once `a + b` passed `5.64e102`: `n·n·(n+1)`, `√(a·b)` and the variance at an extreme
ratio each left `f64` while the values they encode stayed inside it. The Lambert W defect's shape
exactly — see [the defects](#almost-none-of-them-was-a-bug--and-then-three-were).

**The interesting part is why it was not already caught**, because the premise behind that "47"
was wrong. `identities::beta_range` *did* reach this branch, and on the sharpest instrument
available: it sweeps `I_½(a, a) = ½` — an exact identity, no reference to transcribe — over
`a = 1e7 … 1e100`, and checks the complement identity at pairs to `(1e10, 1e10)`. The branch had
been executing all along.

It stopped at `1e100`, and the overflow begins at `2.82e102`. **A round number is not a
boundary**: `1e100` reads as "the whole range" and leaves 206 decades of `f64` behind it, which
is where the entire defect lived. The sibling `gamma_large_shape` sweeps to `1e300` and has no
such gap — the two files were written from the same template and chose differently, and only one
choice reached the top of the format.

The second gap was `γ₁`. It vanishes identically at `a = b`, so a symmetric sweep never evaluates
it, and the complement identity is satisfied by *any* antisymmetric `γ₁` because swapping the
shapes negates `z` and `γ₁` together — constraining the sign and not the magnitude. So the
skewness correction, the one term distinguishing this branch from a plain normal, had never been
compared against an independent value.

Closed by `identities::beta_large_shape`, against 25 out-of-band `Decimal` references computed
inside the branch. Those were affordable because the all-positive `2F1` series needs only
`≈21√a` terms — 68,000 at `a = 1e7`, not the `O(a)` a first estimate suggested. Worth
remembering the next time an in-branch reference looks impossible.

Measured `6.05e-9` worst absolute error against the reference, which is also what the kernel's
own crossover derivation predicts (`≈0.0605/min(a,b)`). The file now stands at **224 mutants,
204 caught, 19 survivors, 1 unviable, no timeouts** — a complete run, against the 47 this table
used to list.

The 19 are in three groups, and none of them is an accuracy case that could be added:

| Where | Count | Why it survives |
|---|---|---|
| `beta_edgeworth`, `n ± 1` and `n ± 2` in `σ` and `γ₁` | 5 | moves the result `3e-11`–`6e-9`, at or under the method's own error |
| `beta_edgeworth`, `z > 0.0` → `>=` | 1 | the line is only reached when `|z| > 40`, so `z = 0` cannot occur |
| `beta_reg`, the branch-selection comparison at `x ≈ (a+1)/(a+b+2)` | 2 | both sides are correct and the code complements; moving the switch changes which is taken |
| `beta_cf`, the Lentz `|d| < 1e-300` renormalisations | 11 | reachable only if `aa·d` equals `−1` to ~300 digits |

The first group is the failure this file already records as "a tolerance loose enough for the real
code can be loose enough for the mutant", and it is unusually clean here: the perturbation is
*smaller* than the Edgeworth truncation error, so an accuracy bound that admits the real kernel
admits the mutant by construction. The second is provably unreachable. The third is the same
crossover equivalence as `erf`/`erfc`/`erfcx`. The fourth is a defensive guard against a case no
realistic input produces — the `within_relative` and `condition_at` family — and is **not
established equivalent**: that would need the brute-force differential search this file requires
before an exclusion, and none has been run. Left visible rather than excluded.

#### An operational note on running this

Both attempts at a full `beta.rs` run died with `Disk quota exceeded (os error 122)` — at mutant
100 with four workers, then at 223 of 224 with two. cargo-mutants builds each scenario in a
scratch copy under `TMPDIR`, and `/tmp` here is a quota'd tmpfs. `TMPDIR` pointed at a roomy
filesystem fixes it; `CARGO_PROFILE_DEV_DEBUG=0` and `--jobs` help but did not suffice. Worth
setting before a long run rather than after, since a run that dies at 99% is not evidence — this
file's own rule — and the single missing mutant had to be re-run with `--re` to complete it.

#### The quantiles: 56 survivors and no defect behind them

Listed as "the f64 seeds, same cause-6 problem as their f32 peers" — the suspicion being that
`gamma_lr_inv` and `beta_reg_inv` carried something like the f32 `erf_inv` Newton step, whose 34
survivors turned out to be a correction that destroyed its own seed's precision.

**They do not, and the structure is why.** Measured against 112 independent 120-digit `Decimal`
bisections, euclid agrees to `1.1e-13` (gamma) and `1.3e-14` (beta) relative. The survivors are
cause 8 below, not cause 6, and the two are worth telling apart because they call for opposite
responses: cause 6 was a design defect worth deleting code over, and this is a design *guarantee*
worth leaving alone.

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

## `euclid-algebra` — 13, now 0

All thirteen were in `conformance.rs`, all the same vacuity pattern: a law suite replaceable with
an empty body, because every test ran it against a *correct* carrier. `star_laws`,
`canonical_order_laws`, `ordered_semiring_laws`, `left_semimodule_laws`, the two `witness_*`, and
the `generated::*` family with its `settings`/`cases`. A focused run over that file is now 53
mutants, 42 caught, 11 unviable, **0 missed**.

The template is `src/tests/vacuity.rs` — a carrier deliberately wrong in exactly one law, claiming
the marker it violates, with the suite required to panic. Cheap on the semiring trunk, where the
laws are marker traits; more work above it, where a tier has to be carried before it can be broken
in one corner. `Faulty<0..4>` shows the pattern for that.

### Two of these laws cannot be broken from outside

Closing them turned up a ceiling on what a vacuity carrier can reach, and it is the reason these
four outlasted the semiring suites. Both are recorded on
`tests/vacuity/order_star_module.rs` in full; the short version:

- **`CanonicalOrder::precedes` is blanket-implemented**, so no carrier can override it. The
  suite's central assertion, `a.precedes(b) == (a.add(b) == a)`, compares the default body with
  its own definition — a tautology, true for everything that can exist. What remains falsifiable
  is reflexivity, antisymmetry and transitivity, which are properties of `⊕`.
- **Monotone `⊗` is implied by distributivity.** `a ⊕ b = a` with right-distributivity gives
  `a⊗c = a⊗c ⊕ b⊗c`, which *is* `a⊗c ⪯ b⊗c` — and `semiring_laws` already checks both
  distributivity directions on the same sample. So nothing can pass `dioid_laws` and fail
  `ordered_semiring_laws`'s extra assertions.

Neither is a reason to delete anything: both are public API a consumer runs, and an assertion
that names its axiom documents the tier better than a comment would. But the *bodies* are still
killable, because an empty body also skips the `dioid_laws` call these suites open with — which
is what the tests exploit. Worth knowing before trying to write a carrier that cannot exist.

### The two knobs behind the generated suites

`GeneratedSample::cases` could return `1` instead of 300, and `generated::settings` could return
`Config::default()` — silently substituting `Config`'s own 256 for whatever the carrier declared.
Neither is reachable by a rejection test: both change how *hard* the search looks, not whether a
broken carrier is caught, and a broken carrier fails on its first case either way. Counting
generated draws would make the assertion statistical, which this repository does not accept for a
property, so `settings` is `pub(crate)` and the two values are asserted directly. A visibility
widening is the cheaper price.

## Historical `seam_ratio` survivors

Three equivalent mutants from the earlier audit cannot be excluded by regex without hiding a real finding in the same
function, since `exclude_re` matches a mutation's description and not its line. The reasoning lives
on `seam_ratio` in the source, where someone reading a report will look: both `||`s in its
finiteness guard are early-outs the closing `ratio.is_finite()` makes redundant, and
`x + window == x` mutated to `x - window == x` duplicates the clause on the next line.

**Do not "fix" these.** The other mutations of the same guard are real and are pinned — at
`x = -f64::INFINITY` three of them make the function report a finite ratio of zero for an argument
where nothing was measured.

## Three notes on choosing the instrument

The `euclid-scalar` work turned on picking the right comparison, more than on writing more
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
