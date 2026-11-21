//! The `erfcx` continued fraction's **depth ladder**, and the search that sized it.
//!
//! A child module so the parent stays inside the crate's 400-line budget, and a natural seam:
//! the fraction is evaluated backward from a fixed depth with no convergence test, so this table
//! *is* the accuracy of `erfc` in its tail. The derivation that sets each number is longer than
//! the code it governs, which is the signature of a file of its own.

/// How many levels the backward recurrence needs at this magnitude, in `f64`.
///
/// # The depth is the accuracy
///
/// The fraction is evaluated **backward from a fixed depth** with no convergence test, so its
/// accuracy is set entirely by that depth, and the depth needed falls steeply as `x` grows.
/// Measured, the levels required for half an ulp:
///
/// ```text
/// x        1.0  1.25  1.375  1.5  1.75   2   2.5   3    4    5    7   10   15   20   50
/// levels   220   160    120  102    78  62    44  34   24   18   14   10    8    8    6
/// ```
///
/// A single fixed depth has to be the worst case, so `120` everywhere was 20x the work needed at
/// `x = 10` and 12x at `x = 5` — paid on every call. Bucketing by magnitude keeps the accuracy
/// where it is scarce and returns the rest.
///
/// # The buckets are sized by a no-regression search, not by a rule of thumb
///
/// They used to sit at "roughly 1.3x the measured requirement at the left edge", which is a
/// reasonable heuristic and was leaving a great deal on the table — because the requirement falls
/// steeply *inside* a bucket as well as between them. One bucket covering `[ERFCX_CF_MIN, 2)` at
/// 160 levels was serving a requirement that runs from 108 at its left edge to 54 at its right.
///
/// So each bucket's depth is now the smallest that is **no worse than the previous depth over
/// that same band**, measured against a 2,000-level reference across 20,000 points per band:
///
/// ```text
/// band          [1.375,1.5) [1.5,1.75) [1.75,2) [2,3) [3,5) [5,10) [10,20) [20,50) [50,∞)
/// requirement           108         98       73    63    30     16       9       6      4
/// was                   160        160      160    72    44     24      14      10     10
/// now                   145        131      102    72    44     20      11       7      5
/// worst ulp, was      0.740      0.000    0.000 0.997 0.000  0.000   0.000   0.000  0.000
/// worst ulp, now      0.736      0.000    0.000 0.997 0.000  0.000   0.000   0.000  0.000
/// ```
///
/// Every band is at least as accurate as it was — `[1.375,1.5)` is fractionally *better* — while
/// the work falls by 9% to 50%. Two bands do not move at all, and that is the search working
/// rather than failing: `[2,3)` is the loosest in the ladder at `0.997` ulp and has nothing to
/// give, and `[3,5)` is already minimal.
///
/// The far tail matters more than its width suggests, because it is where the normal log-CDF
/// spends its time: `log_ndtr` at a large negative argument is `erfcx` at a large positive one.
/// `[50,∞)` needed 4 levels and was being given 10.
///
/// Depth mattered more than it looked. At the `60` this used to be, the fraction carried `25.6`
/// ulp at `ERFCX_CF_MIN` — the loosest point in the whole `erfc` family — and the threshold had
/// been *placed* against that, two tenths too high. Deepening it fixed the crossover and let
/// `ERFCX_CF_MIN` come down with it.
///
/// # Mutation testing is what asked for this
///
/// Every comparison in the ladder survived. They are not equivalent mutants — each moves a band
/// of arguments onto a neighbouring depth — but no *accuracy* assertion can see it, because the
/// buckets are deliberately sized above their requirement, so one bucket over still converges.
/// What is worth pinning is the decision, and only a named decision can be asserted directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ErfcxDepth {
    /// `[ERFCX_CF_MIN, 3/2)`, where the requirement peaks and the ladder is tightest.
    UnderThreeHalves,
    /// `[3/2, 7/4)`.
    UnderSevenQuarters,
    /// `[7/4, 2)`.
    UnderTwo,
    /// `[2, 3)` — the loosest band in the ladder, and the one with no slack to return.
    UnderThree,
    /// `[3, 5)`, already minimal.
    UnderFive,
    /// `[5, 10)`.
    UnderTen,
    /// `[10, 20)`.
    UnderTwenty,
    /// `[20, 50)`.
    UnderFifty,
    /// `[50, ∞)`, the floor — and the band the normal log-CDF's tail spends its time in.
    AtLeastFifty,
}

impl ErfcxDepth {
    /// The magnitude selects the bucket.
    pub(super) fn for_argument(x: f64) -> Self {
        if x < 1.5 {
            Self::UnderThreeHalves
        } else if x < 1.75 {
            Self::UnderSevenQuarters
        } else if x < 2.0 {
            Self::UnderTwo
        } else if x < 3.0 {
            Self::UnderThree
        } else if x < 5.0 {
            Self::UnderFive
        } else if x < 10.0 {
            Self::UnderTen
        } else if x < 20.0 {
            Self::UnderTwenty
        } else if x < 50.0 {
            Self::UnderFifty
        } else {
            Self::AtLeastFifty
        }
    }

    /// Levels for this bucket: the smallest depth no worse than the previous one over the same
    /// band, measured against a 2,000-level reference. See this type's documentation for the
    /// table and for the two bands that could not be reduced.
    pub(super) fn levels(self) -> u32 {
        match self {
            Self::UnderThreeHalves => 145,
            Self::UnderSevenQuarters => 131,
            Self::UnderTwo => 102,
            Self::UnderThree => 72,
            Self::UnderFive => 44,
            Self::UnderTen => 20,
            Self::UnderTwenty => 11,
            Self::UnderFifty => 7,
            Self::AtLeastFifty => 5,
        }
    }
}
