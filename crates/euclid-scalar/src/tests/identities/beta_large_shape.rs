//! The incomplete beta at large shape, where its continued fraction hands over.
//!
//! The peer of [`super::gamma_large_shape`], past the threshold where the continued fraction is
//! abandoned for an Edgeworth-corrected normal.
//!
//! # What was already covered, and what was not
//!
//! Worth stating precisely, because the gap here was not the obvious one.
//! [`super::beta_range`] already reached this branch: it sweeps `I_½(a, a) = ½` over
//! `a = 1e7 … 1e100` and checks the complement identity at pairs up to `(1e10, 1e10)`. The
//! branch executed, and on an exact identity. Two things were still missing:
//!
//! - **The top of the exponent range.** That sweep stops at `1e100`. The format goes to
//!   `1.8e308`, and `1e100` is a round number rather than a boundary — it reads as "the whole
//!   range" and leaves 206 decades of shape behind it. The defect below was in all of them.
//! - **`γ₁` at an asymmetric shape.** The skewness correction vanishes identically when `a = b`,
//!   so the symmetric sweep never evaluates it. The complement identity does run asymmetric
//!   pairs, but the Edgeworth form satisfies that identity for *any* antisymmetric `γ₁` —
//!   swapping the shapes negates `z` and `γ₁` together — so it constrains the term's sign and
//!   not its magnitude. Nothing had compared `γ₁` against an independent value.
//!
//! # The defect this found
//!
//! `I_x(a, b)` returned **NaN for every `x`** once `a + b` passed `5.64e102` — so for `a = b`
//! above `2.82e102`, two and a half decades past where the existing sweep stopped. The answers
//! it was failing to produce were not delicate: `I_½(a, a)` is exactly `½` at every shape, by
//! symmetry of `Beta(a, a)`. Three intermediates left `f64` while the values they encode stayed
//! well inside it:
//!
//! | Intermediate | Leaves `f64` at | What it encodes | True value there |
//! |---|---|---|---|
//! | `n·n·(n+1)` | `n > 5.6e102` | `σ` at `a = b = 1e150` | `3.5e-76` |
//! | `a·b` | `a·b > 1.8e308` | `γ₁` | `0` for `a = b` |
//! | the variance | `a = f64::MAX, b = 1e7` | `σ` | `1.8e-305` |
//!
//! Each is now formed from bounded ratios, and `z` is short-circuited once the normal tail has
//! saturated — which is also what stops `z·z` overflowing into `0 · ∞ = NaN`.
//!
//! The gamma peer never had this, because its variance is `a` rather than a quotient of three
//! `O(n)` factors. That asymmetry is why the beta needed its own file rather than a few more
//! rows in an existing one.
//!
//! # What the references are
//!
//! Three independent kinds, none of them euclid:
//!
//! - **An exact closed form.** `Beta(a, a)` is symmetric about `½`, so `I_½(a, a) = ½`
//!   identically. No approximation, no table, true at every `a` — and it is what the overflow
//!   broke.
//! - **60-digit `Decimal` values**, generated out-of-band from the all-positive series
//!   `I_x(a,b) = x^a (1−x)^b / (a·B(a,b)) · Σ (a+b)ₙ/(a+1)ₙ xⁿ`, which has no subtraction in it
//!   and so no cancellation. Its terms decay like `e^{−n²/4a}`, needing about `21√a` of them —
//!   68,000 at `a = 1e7`, which is why an in-branch reference is affordable at all. The
//!   generator was checked against `I_x(1,1) = x`, `I_x(a,1) = xᵃ`, `I_x(1,b) = 1−(1−x)ᵇ` and
//!   the exact polynomial `I_x(2,3)` before any value here was taken from it.
//! - **The converged continued fraction**, just below the threshold. Wholly unrelated to a
//!   normal CDF with a skewness correction, so agreement across the join is real evidence the
//!   join is sited correctly rather than a restatement.
//!
//! # Each `x` is written as the double the reference was evaluated at
//!
//! The density is `≈ 3568` at `a = b = 1e7`, so one ulp of `x` is worth `4e-13` of probability —
//! measured up to `4.4e-13` across this table. Computing a reference at the real-valued
//! `μ + zσ` and comparing it against euclid's value at the *rounded* `x` would charge the kernel
//! for that rounding, which is `44×` its actual error here.
//!
//! So each `x` below is in its shortest round-tripping decimal form, and the reference beside it
//! was computed from `Decimal(x)` — exact for a double — rather than from the expression `x` came
//! from. The two columns therefore describe the same argument exactly.

use crate::Real;
use euclid_utils::{assert_close, assert_exact};

use crate::kernels::beta::incomplete::BETA_EDGEWORTH_MIN;

/// `(a, b, x, I_x(a,b))`, the `Decimal` references described in the module header.
///
/// **Regenerate with `python3 -B bin/references/incomplete.py beta-large-shape`**, which prints
/// these rows verbatim. The generator validates itself against `I_x(1,1) = x`, `I_x(a,1) = xᵃ`,
/// `I_x(1,b) = 1−(1−x)ᵇ`, the exact polynomial `I_x(2,3)` and `I_½(a,a) = ½` before emitting
/// anything, and exits non-zero if any of those fail. It is committed for the same reason
/// `bin/mutants.py` is: a reference whose instrument is missing is a number taken on faith.
///
/// Both symmetric and asymmetric shapes, because they test different code: `γ₁` is
/// `2(b−a)√(n+1)/((n+2)√(ab))`, which **vanishes identically when `a = b`**. A suite of
/// symmetric cases alone would leave the entire skewness correction unexercised, and it is the
/// one term that distinguishes this branch from a plain normal approximation.
const EDGEWORTH_REFERENCE: &[(f64, f64, f64, f64)] = &[
    // a = b: skewness is exactly zero, so these pin the mean/variance path alone.
    (1e7, 1e7, 0.49966458981176026, 0.0013498970344633411),
    (1e7, 1e7, 0.4998881966039201, 0.15865525998070798),
    (1e7, 1e7, 0.5, 0.5),
    (1e7, 1e7, 0.5001118033960799, 0.8413447400191719),
    (1e7, 1e7, 0.5003354101882397, 0.9986501029655345),
    // Ten to one, both orientations: `γ₁` is now ±5.4e-4 and must carry its sign.
    (1e7, 1e8, 0.09082686054276165, 0.001346692639660199),
    (1e7, 1e8, 0.09088168078698115, 0.15865525206114425),
    (1e7, 1e8, 0.09090909090909091, 0.5000360856871126),
    (1e7, 1e8, 0.09093650103120066, 0.8413447479377909),
    (1e7, 1e8, 0.09099132127542016, 0.9986468933492093),
    (1e8, 1e7, 0.9090086787245798, 0.001353106650783902),
    (1e8, 1e7, 0.9090634989687993, 0.15865525206220912),
    (1e8, 1e7, 0.9090909090909091, 0.4999639143124835),
    (1e8, 1e7, 0.9091183192130189, 0.8413447479392233),
    (1e8, 1e7, 0.9091731394572383, 0.9986533073603353),
    // A hundred to one, and a non-decade pair, so the grid is not all powers of ten.
    (1e7, 1e9, 0.00989164381026907, 0.0013462188497053026),
    (1e7, 1e9, 0.009897874669429624, 0.15865525013693643),
    (1e7, 1e9, 0.009900990099009901, 0.5000414250765153),
    (1e7, 1e9, 0.009904105528590178, 0.8413447498613683),
    (1e7, 1e9, 0.009910336387750733, 0.9986464180952248),
    (2e7, 5e7, 0.2855523009412959, 0.0013485579952529215),
    (2e7, 5e7, 0.2856602907899558, 0.15865525514140152),
    (2e7, 5e7, 0.2857142857142857, 0.5000150786007935),
    (2e7, 5e7, 0.28576828063861565, 0.841344744858726),
    (2e7, 5e7, 0.2858762704872755, 0.9986487618750133),
];

/// Measured worst absolute error `6.05e-9` over [`EDGEWORTH_REFERENCE`].
///
/// Absolute rather than relative, and `assert_close`'s mixed criterion makes it so for free
/// here: every reference is a probability, so `max(|want|, 1)` is 1. That is the right bar for
/// a CDF — a relative bound would demand `1e-9` of a `1.3e-3` tail value, which is `7.4e-7`
/// relative, and nothing about this approximation claims that.
///
/// It is also the error the kernel's own crossover analysis predicts: `≈ 0.0605/min(a,b)`, or
/// `6.1e-9` at `min = 1e7`. The agreement between that derivation and this measurement is the
/// substance of the check — an independent reference confirming the documented error model,
/// rather than a tolerance fitted to whatever came out.
const EDGEWORTH_TOL: f64 = 1e-8;

mod identities;
mod references;
mod resolution;
