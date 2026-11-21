//! Temme's uniform asymptotic expansion for the incomplete gamma at large shape.
//!
//! # The two problems this exists to solve are the same problem
//!
//! Near `x = a` the lower series `Σ x^k/[a(a+1)⋯(a+k)]` has term ratio `x/(a+k) ≈ 1 − k/a`, so
//! its terms decay like `e^{−k²/2a}` and it needs about `8.5√a` of them. That is one counted
//! fact with two consequences, and the benchmark showed both: **32200 terms** at the top of the
//! range, and an accumulated rounding error that tracks `√a` — measured at `0.5√a` ulp, which is
//! `5` ulp at `a = 100` and `104` at `a = 30000`. Making the series shorter and making it more
//! accurate are the same change.
//!
//! An expansion in `1/a` replaces it with a fixed ten terms, independent of `a`.
//!
//! # The expansion, derived rather than quoted
//!
//! With `t = aτ` and `λ = x/a`, `Q(a,x) = (a^a e^{−a}/Γ(a))·∫_λ^∞ e^{−a(τ−1−ln τ)} dτ/τ`.
//! Substituting `½η′² = τ − 1 − ln τ` with `sign(η′) = sign(τ−1)` turns the measure into
//! `dτ/τ = η′dη′/(τ−1)`, giving
//!
//! ```text
//! Q = √(a/2π)/Γ*(a) · ∫_η^∞ e^{−aη′²/2}·(η′/σ(η′)) dη′,     σ = τ − 1
//! ```
//!
//! Splitting the integrand as `1 + h`, the `1` integrates to `erfc` exactly. The rest is expanded
//! by integration by parts — but **uniformly**: at each stage the value at `η′ = 0` is split off
//! before integrating, because the naive recursion divides by `η` and produces poles at the
//! origin, which is precisely where the expansion is needed. The split-off values collect into
//! `Γ*(a)` and cancel the prefactor, leaving
//!
//! ```text
//! Q(a,x) = ½erfc(sign(σ)√y) + e^{−μ(a)}·(e^{−y}/√(2πa))·Σ_k d_k(η)/a^k
//! ```
//!
//! with `φ = σ − ln(1+σ)`, `y = aφ`, `η = sign(σ)√(2φ)`, and `μ` the Stirling remainder the
//! `Γ` kernel already carries — `Γ*(a) = e^{μ(a)}` exactly.
//!
//! That the `d_k` recursion's discarded constants reproduce Γ*'s Stirling series is not an
//! input to the derivation; it is what tells us the derivation is right. See [`TEMME_D`].
//!
//! # What it does not fix
//!
//! Where the answer is exponentially small the floor is `y` ulp, because `e^{−y}` cannot be
//! computed to better than `y·ε` relative when `y` is only known to `y·ε` absolute. At
//! `a = 10000`, `σ = 0.15` that is `102` ulp and no number of terms moves it. The expansion is
//! correctly rounded where the answer is `O(1)`, which is the region that carries the mass.

/// The `d_k(η)` of the uniform expansion, as Taylor coefficients in `η`, lowest first.
///
/// Derived, not transcribed. `temme_coeffs.py` builds them as exact rationals by reverting
/// `½η² = σ − ln(1+σ)` and running the uniform integration-by-parts recursion
/// `d_k = (h_k − h_k(0))/η`, `h_{k+1} = d_k′`, then emits this table. The derivation checks
/// itself: the values split off at `η = 0` come out as Γ*'s Stirling series
/// (`1/12`, `1/288`, `−139/51840`, `−571/2488320`, `163879/209018880`), which the
/// construction never supplies — that is the `erfc` multiplier collecting to Γ* and
/// cancelling the prefactor, confirmed rather than assumed.
///
/// Each row is truncated against **its own** weight: `d_k` is divided by `a^k ≥ 20^k`, so
/// the tenth row needs three terms where the first needs nineteen. Sizing every row for the
/// first row's precision would be 204 coefficients instead of 124.
const TEMME_D: [&[f64]; 10] = [
    // d_0
    &[
        -0.3333333333333333,
        0.08333333333333333,
        -0.014814814814814815,
        0.0011574074074074073,
        0.0003527336860670194,
        -0.0001787551440329218,
        3.919263178522438e-05,
        -2.185448510679992e-06,
        -1.85406221071516e-06,
        8.296711340953087e-07,
        -1.7665952736826078e-07,
        6.707853543401498e-09,
        1.0261809784240309e-08,
        -4.382036018453353e-09,
        9.14769958223679e-10,
        -2.5514193994946248e-11,
        -5.830772132550426e-11,
        2.4361948020667415e-11,
        -5.0276692801141755e-12,
    ],
    // d_1
    &[
        -0.02962962962962963,
        0.003472222222222222,
        0.0014109347442680777,
        -0.000893775720164609,
        0.00023515579071134627,
        -1.5298139574759944e-05,
        -1.483249768572128e-05,
        7.467040206857778e-06,
        -1.766595273682608e-06,
        7.378638897741648e-08,
        1.231417174108837e-07,
        -5.696646823989359e-08,
        1.2806779415131507e-08,
        -3.8271290992419376e-10,
        -9.32923541208068e-10,
        4.141531163513461e-10,
        -9.049804704205516e-11,
    ],
    // d_2
    &[
        0.0028218694885361554,
        -0.0026813271604938273,
        0.0009406231628453851,
        -7.649069787379973e-05,
        -8.899498611432768e-05,
        5.226928144800444e-05,
        -1.4132762189460864e-05,
        6.640775007967483e-07,
        1.231417174108837e-06,
        -6.266311506388295e-07,
        1.536813529815781e-07,
        -4.975267829014519e-09,
        -1.3060929576912952e-08,
        6.212296745270191e-09,
        -1.4479687526728825e-09,
    ],
    // d_3
    &[
        0.0018812463256907702,
        -0.00022947209362139917,
        -0.0003559799444573107,
        0.0002613464072400222,
        -8.479657313676519e-05,
        4.6485425055772385e-06,
        9.851337392870696e-06,
        -5.639680355749465e-06,
        1.5368135298157807e-06,
        -5.47279461191597e-08,
        -1.5673115492295543e-07,
        8.075985768851248e-08,
        -2.0271562537420356e-08,
        5.331627939482747e-10,
        1.9421356391429678e-09,
    ],
    // d_4
    &[
        -0.0007119598889146215,
        0.0007840392217200666,
        -0.00033918629254706074,
        2.3242712527886193e-05,
        5.9108024357224175e-05,
        -3.947776249024626e-05,
        1.2294508238526246e-05,
        -4.925515150724373e-07,
        -1.5673115492295543e-06,
        8.883584345736373e-07,
        -2.432587504490443e-07,
        6.931116321327572e-09,
        2.7189898948001546e-08,
        -1.416685905624359e-08,
    ],
    // d_5
    &[
        -0.0006783725850941215,
        6.972813758365857e-05,
        0.0002364320974288967,
        -0.0001973888124512313,
        7.376704943115748e-05,
        -3.4478606055070616e-06,
        -1.2538492393836434e-05,
        7.995225911162736e-06,
        -2.432587504490443e-06,
        7.624227953460329e-08,
        3.2627878737601855e-07,
        -1.8416916773116666e-07,
    ],
    // d_6
    &[
        0.0004728641948577934,
        -0.0005921664373536939,
        0.0002950681977246299,
        -1.7239303027535307e-05,
        -7.523095436301861e-05,
        5.596658137813915e-05,
        -1.9460700035923543e-05,
        6.861805158114295e-07,
        3.2627878737601857e-06,
        -2.0258608450428333e-06,
        6.074782128624808e-07,
    ],
    // d_7
    &[
        0.0005901363954492598,
        -5.171790908260592e-05,
        -0.00030092381745207443,
        0.0002798329068906958,
        -0.00011676420021554124,
        4.803263610680007e-06,
        2.6102302990081485e-05,
        -1.82327476053855e-05,
        6.074782128624807e-06,
    ],
    // d_8
    &[
        -0.0006018476349041489,
        0.0008394987206720873,
        -0.000467056800862165,
        2.4016318053400035e-05,
        0.0001566138179404889,
        -0.0001276292332376985,
        4.859825702899846e-05,
    ],
    // d_9
    &[
        -0.00093411360172433,
        7.204895416020011e-05,
        0.0006264552717619556,
        -0.0006381461661884925,
        0.00029158954217399074,
    ],
];

/// At or above this shape the uniform expansion replaces the series.
///
/// `20`, which is [`STIRLING_MIN`](crate::math::gamma::STIRLING_MIN) — not a coincidence, since
/// the expansion needs `μ(a)` and that series is only valid from there. Measured against a
/// 400-bit reference with ten orders kept: `≤ 2` ulp at `a = 20`, and the truncation falls away
/// as `a^{-10}` above it.
const TEMME_MIN_SHAPE: f64 = 20.0;

/// The widest `|σ| = |x−a|/a` the expansion is used on.
///
/// `0.4`, chosen from both sides. Below it the series is the expensive one — it is long only
/// while `|σ| ≲ 1/√a`, and by `|σ| = 0.4` it converges geometrically in about ninety terms
/// whatever `a` is, so there is nothing left to win. Above it the `d_k` would need half as many
/// coefficients again: the table is sized at `|η| ≤ 0.4707`, which is what `|σ| ≤ 0.4` maps to.
const TEMME_MAX_OFFSET: f64 = 0.4;

/// Which tail the expansion returns without cancelling.
///
/// `½erfc(√y)` is the *small* tail either side of `x = a`, and the correction's sign is what
/// distinguishes them. Taking the large tail and subtracting from one would throw away exactly
/// the digits this route exists to keep, so the side is a decision with a name rather than a
/// sign buried in an expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TemmeTail {
    /// `x ≥ a`: the upper tail `Q` is the small one.
    Upper,
    /// `x < a`: the lower tail `P` is the small one.
    Lower,
}

impl TemmeTail {
    fn for_arguments(a: f64, x: f64) -> Self {
        if x >= a { Self::Upper } else { Self::Lower }
    }
}

/// Whether the uniform expansion covers this `(a, x)`.
pub(crate) fn use_temme(a: f64, x: f64) -> bool {
    a >= TEMME_MIN_SHAPE && ((x - a) / a).abs() <= TEMME_MAX_OFFSET
}

/// `Σ_k d_k(η)/a^k`, by Horner in `1/a` with each `d_k` by Horner in `η`.
fn temme_sum(eta: f64, a: f64) -> f64 {
    let mut sum = 0.0;
    for row in TEMME_D.iter().rev() {
        let mut term = 0.0;
        for &coefficient in row.iter().rev() {
            term = term * eta + coefficient;
        }
        sum = sum / a + term;
    }
    sum
}

/// The smaller of the two tails at `(a, x)`, evaluated directly.
fn temme_small_tail(a: f64, x: f64) -> f64 {
    let offset = (x - a) / a;
    // `φ = σ − ln(1+σ)`, cancellation-free through the same series the prefactor uses. It is
    // non-negative, and zero only at `x = a`.
    let rate = -crate::math::gamma::ln_1p_minus_linear(offset);
    let scaled = a * rate;
    let eta = (2.0 * rate).sqrt().copysign(offset);

    let correction = (-crate::math::gamma::stirling_remainder(a)).exp() * (-scaled).exp()
        / (crate::constants::f64::SQRT_TAU * a.sqrt())
        * temme_sum(eta, a);

    let half_erfc = 0.5 * crate::kernels::erf::erfc(scaled.sqrt());
    match TemmeTail::for_arguments(a, x) {
        TemmeTail::Upper => half_erfc + correction,
        // `½erfc(−√y) = 1 − ½erfc(√y)`, so `P = 1 − Q` folds to this with no subtraction of
        // nearly equal numbers anywhere.
        TemmeTail::Lower => half_erfc - correction,
    }
}

/// `P(a, x)` by the uniform expansion.
pub(crate) fn temme_lower(a: f64, x: f64) -> f64 {
    match TemmeTail::for_arguments(a, x) {
        TemmeTail::Upper => 1.0 - temme_small_tail(a, x),
        TemmeTail::Lower => temme_small_tail(a, x),
    }
}

/// `Q(a, x)` by the uniform expansion.
pub(crate) fn temme_upper(a: f64, x: f64) -> f64 {
    match TemmeTail::for_arguments(a, x) {
        TemmeTail::Upper => temme_small_tail(a, x),
        TemmeTail::Lower => 1.0 - temme_small_tail(a, x),
    }
}

#[cfg(test)]
mod tests;
