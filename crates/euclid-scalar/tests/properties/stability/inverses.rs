//! Backward error of the `f64` inverses: substitute the answer back.
//!
//! Split from [`super::cross_precision`] on the line that matters. Those properties have a wider
//! peer to compare against, so their bound is the wide result rounded to `f32`. These have no
//! wider peer and no closed form, so the only accuracy statement available is the **backward**
//! error — run the answer back through the forward function and see where it lands.
//!
//! That turns out to be the better question for an inverse anyway. `ndtri` deliberately stops on
//! a two-cycle, so its result can move by up to 18 ulp depending on where the iteration is cut;
//! the backward error does not move, because the adjacent values are all roots. A forward ulp
//! comparison against a "reference" implementation would report that 18 ulp as error. This
//! reports it as the non-uniqueness it is.
//!
//! Three of the four bounds here are the *resolution of the answer* rather than a tolerance —
//! one ulp of the root spans some amount of the forward function, and no implementation can beat
//! half of that. See `BETA_INV_SPAN_SAFETY` and `LAMBERT_W_SPAN_SAFETY`.

use euclid_scalar::Real;
use euclid_stability::{Config, check, classify, generate, property, target};

fn settings(suite: &str) -> Config {
    Config::default()
        .cases(600)
        .regressions(format!("tests/regressions/{suite}.txt"))
        .budget("tests/budgets.txt")
}

// ---- backward error of the inverses ------------------------------------
//
// An inverse has an accuracy measure the forward functions do not: substitute the answer back.
// `|Phi(x) - p| / p` needs no reference more accurate than `Phi` itself, which is why it is the
// one way to state an `f64` inverse's accuracy under this repository's no-bignum rule — every
// other suite here is reduced to identities or conditioning arguments.
//
// It also measures the right thing. `ndtri` deliberately stops on a two-cycle, so its result can
// move by up to 18 ulp depending on where the iteration is cut; the backward error does not
// move, because the adjacent values are all roots. A forward ulp comparison against a
// "reference" implementation would report that 18 ulp as error. This reports it as the
// non-uniqueness it is.

/// Base term of the backward-error allowance, in ulps of `p`. The `x²` term is added per point.
///
/// # The allowance has to carry `x²`, and that is conditioning rather than slack
///
/// A flat bound is the wrong shape here. `p = Phi(x)` has `d(ln p)/dx = phi(x)/Phi(x) ≈ |x|` in
/// the lower tail, so **one ulp of `x` moves `p` by `ε·x²` relatively** — the inverse cannot do
/// better than that however it is written, because the answer is only representable to `ulp(x)`.
/// At `p = 1e-300` the root is `x = -37.05`, giving `ε·x² = 3.0e-13`; the measured backward error
/// there is `1.0e-13`, comfortably inside a bound it cannot escape.
///
/// A flat `64·ε` bound was tried first and fails at exactly that point. It is the same `x²`
/// structure as `erfc`'s allowance, arrived at from the opposite direction: there `exp(−x²)`
/// turns an absolute error in the exponent into a relative one, here `Phi`'s logarithmic
/// derivative turns an ulp of the root into a relative error in `p`.
///
/// `256` leaves roughly fourfold headroom over the `8.9e-15` worst case in the central region.
const NDTRI_BACKWARD_BASE: f64 = 256.0;

property! {
    #![config(settings("f64_ndtri_backward")
        .require("central", 20.0)
        .require("tail", 20.0))]

    /// `Phi(ndtri(p))` returns `p`, to the accuracy `Phi` itself can deliver.
    ///
    /// Both halves of the distribution are required, because `ndtri` seeds them differently —
    /// `central_start` through `erf_inv` above the cutoff, the log-asymptotic form below it — and
    /// a sweep that reached only one would say nothing about the other.
    fn the_normal_quantile_inverts_its_own_cdf(p in generate::float::in_range(1e-300, 0.5)) {
        if !(p > 0.0 && p < 0.5) {
            return Ok(());
        }
        classify!("central", p >= 1e-15);
        classify!("tail", p < 1e-15);

        let x = Real::ndtri(p);
        if !x.is_finite() {
            return Ok(());
        }
        // `Phi(x)` as `erfc(-x/sqrt 2)/2`, which keeps relative accuracy for a small lower tail
        // rather than losing it to a complement.
        let back = 0.5 * Real::erfc(-x * core::f64::consts::FRAC_1_SQRT_2);
        let backward = ((back - p) / p).abs();
        let allowed = f64::EPSILON * (NDTRI_BACKWARD_BASE + x * x);
        target(backward / allowed);
        check!(
            backward <= allowed,
            "ndtri({p:e}) = {x:e} has backward error {backward:e}, over its conditioned \
             allowance of {allowed:e} — Phi of the returned root is {back:e}"
        );
    }
}

/// Base term for the incomplete-gamma inverse's backward error, in ulps of `p`.
///
/// # The allowance grows with the shape, and that is the root's own resolution
///
/// `gamma_lr_inv` returns `x ≈ a` near the median, so one ulp of the answer is `ε·a` — and the
/// CDF climbs there at the peak density `≈ 1/√(2πa)`, so that ulp is worth `ε·a/√(2πa)` of
/// probability. Against a `p` of order one that is `≈ ε·√a` relative, before the CDF's own error
/// is added. Measured, the growth is steeper than `√a`: `6.9e-16` at `a = 0.5`, `4.9e-16` at
/// `a = 2`, `4.5e-13` at `a = 1e3` and **`9.3e-10` at `a = 1e6`** — close to `ε·a`.
///
/// So the bound carries a term linear in the shape. Nothing in the suite measured this before,
/// and `9.3e-10` is worth knowing about for a function whose forward peer is documented near
/// `1e-14`: inverting a CDF at a large shape is intrinsically lossy, because the answer is a
/// large number and the probability only resolves it to `ulp(a)`.
const GAMMA_INV_BACKWARD_BASE: f64 = 256.0;

/// Shape coefficient for the same allowance; see [`GAMMA_INV_BACKWARD_BASE`].
const GAMMA_INV_BACKWARD_SHAPE: f64 = 64.0;

property! {
    // Floors at half the thinnest coverage over six seeds. The shape generator is markedly
    // seed-sensitive here — the small-shape share ranged 9.8% to 59.3% — so these are set from
    // the minimum rather than from the default seed's split.
    #![config(settings("f64_gamma_inv_backward")
        .require("small shape", 4.0)
        .require("large shape", 20.0))]

    /// `P(a, gamma_lr_inv(p, a))` returns `p`, to the accuracy the shape allows.
    ///
    /// The backward error is the only accuracy statement available for this inverse: there is no
    /// wider float to compare against, and no closed form for a general `(p, a)`. Substituting
    /// the answer back needs nothing but the forward CDF.
    fn the_gamma_quantile_inverts_its_own_cdf(
        p in generate::float::in_range(1e-10, 1.0 - 1e-10),
        a in generate::float::magnitude(-2, 7),
    ) {
        let a = a.abs();
        if !(p > 0.0 && p < 1.0) || a <= 0.0 || !a.is_finite() {
            return Ok(());
        }
        classify!("small shape", a < 1.0);
        classify!("large shape", a >= 1.0);

        let x = Real::gamma_lr_inv(p, a);
        if !x.is_finite() || x == 0.0 {
            return Ok(());
        }
        let back = Real::gamma_lr(x, a);
        if !back.is_finite() || back == 0.0 {
            return Ok(());
        }
        let backward = ((back - p) / p).abs();
        let allowed =
            f64::EPSILON * (GAMMA_INV_BACKWARD_BASE + GAMMA_INV_BACKWARD_SHAPE * a);
        target(backward / allowed);
        check!(
            backward <= allowed,
            "gamma_lr_inv({p:e}, {a:e}) = {x:e} has backward error {backward:e}, over its \
             shape-conditioned allowance of {allowed:e} — P(x, a) is {back:e}"
        );
    }
}

/// Multiple of the one-ulp probability span that `beta_reg_inv` is allowed to miss by.
///
/// # This is the sharp bound, and it needs no reference at all
///
/// An inverse cannot return a better answer than the best representable one. One ulp of the root
/// spans some amount of probability, and if the true root falls between two representable values
/// then the nearest of them is off by up to half that span — however the solver is written.
/// Measuring the span costs two extra forward CDF evaluations and turns "is this accurate?" into
/// "is this as accurate as `f64` permits?", which is the question with an answer.
///
/// Measured worst backward error against that span, over 399 probabilities per shape pair:
///
/// ```text
/// shapes        backward error   one-ulp span   ratio
/// 0.1, 0.1          5.416e-3       1.390e-2      0.39
/// 0.5, 0.5          2.449e-15      1.803e-14     0.14
/// 1e3, 1e3          9.943e-13      2.013e-12     0.49
/// ```
///
/// Every ratio is below `0.5`, which is the statement that the kernel returns the *nearest*
/// representable root rather than merely a nearby one.
///
/// The first row is why this bound and not a shape-scaled one. `Beta(0.1, 0.1)` is so strongly
/// U-shaped that the root for `p = 0.9925` is `x = 1.0` exactly, where a single ulp carries
/// **1.4% of the distribution** — so a backward error of `5.4e-3` there is optimal, not broken.
/// A flat tolerance would call it a catastrophe and a shape-linear one would still be wrong,
/// because the cause is the local slope of the CDF rather than the size of the shape.
const BETA_INV_SPAN_SAFETY: f64 = 4.0;

/// Floor in ulps of `p`, for roots where the span collapses to nothing.
const BETA_INV_FLOOR: f64 = 256.0;

property! {
    #![config(settings("f64_beta_inv_backward")
        .require("small shapes", 10.0)
        .require("large shapes", 10.0))]

    /// `I_x(a, b)` returns `p` for `x = beta_reg_inv(p, a, b)`, to within half an ulp of the root.
    ///
    /// Stated against the representability limit rather than a tolerance, so the claim is "this
    /// inverse is as accurate as the format allows" — see [`BETA_INV_SPAN_SAFETY`].
    fn the_beta_quantile_inverts_its_own_cdf(
        p in generate::float::in_range(1e-6, 1.0 - 1e-6),
        a in generate::float::magnitude(-1, 4),
        b in generate::float::magnitude(-1, 4),
    ) {
        let (a, b) = (a.abs(), b.abs());
        if !(p > 0.0 && p < 1.0) || a <= 0.0 || b <= 0.0 || !a.is_finite() || !b.is_finite() {
            return Ok(());
        }
        classify!("small shapes", a.min(b) < 1.0);
        classify!("large shapes", a.min(b) >= 1.0);

        let x = Real::beta_reg_inv(p, a, b);
        if !(x > 0.0 && x < 1.0) {
            return Ok(());
        }
        let back = Real::beta_reg(x, a, b);
        if !(back.is_finite() && back > 0.0) {
            return Ok(());
        }
        // The probability spanned by the root's two neighbours: the resolution of the answer.
        let up = Real::beta_reg(x.next_up(), a, b);
        let down = Real::beta_reg(x.next_down(), a, b);
        if !(up.is_finite() && down.is_finite()) {
            return Ok(());
        }
        let span = ((up - down) / p).abs();
        let backward = ((back - p) / p).abs();
        let allowed = BETA_INV_SPAN_SAFETY * span + BETA_INV_FLOOR * f64::EPSILON;
        target(backward / allowed);
        check!(
            backward <= allowed,
            "beta_reg_inv({p:e}, {a:e}, {b:e}) = {x:e} has backward error {backward:e}, over the \
             {allowed:e} its own resolution allows — one ulp of the root spans {span:e} of \
             probability, and I_x is {back:e}"
        );
    }
}

/// Multiple of the one-ulp span that `lambert_w` is allowed to miss its defining equation by.
///
/// # `W` has an exact residual, which is rare here
///
/// `W(x)·e^{W(x)} = x` is the definition, so the backward error can be formed with no reference,
/// no closed form and no wider float — just one `exp` and one multiply. That makes `W` the
/// cleanest case in this file: unlike the `f32` kernels there is nothing to compare against, and
/// unlike the quantiles there is no CDF to re-evaluate.
///
/// The bound is again the resolution of the answer rather than a tolerance. One ulp of `W` moves
/// `x` by `|W·e^W|·(1 + 1/W)·ulp(W)`, which the test measures directly from `W`'s neighbours.
/// Measured worst backward error against that span:
///
/// ```text
/// region                     backward error   one-ulp span   ratio
/// at the branch point            1.509e-16      1.509e-16     1.00
/// near the branch point          3.357e-16      1.678e-16     2.00
/// negative                       2.217e-16      2.217e-16     1.00
/// small positive                 3.441e-16      8.030e-16     0.43
/// large                          1.190e-15      3.966e-15     0.30
/// very large  (x = 4.8e299)      5.690e-14      2.278e-13     0.25
/// ```
///
/// The worst *absolute* figure, `5.7e-14` at `x = 4.8e299`, is a quarter of the span there, so it
/// is the best representable answer rather than a loss of accuracy. The ratio peaks at `2` beside
/// the branch point, where `W` has a square-root singularity and the Halley iteration is seeded
/// from a series — so the answer is good to a couple of ulps there and to under one elsewhere.
const LAMBERT_W_SPAN_SAFETY: f64 = 8.0;

/// The branch point `−1/e`, where `dW/dx` is infinite.
const LAMBERT_BRANCH: f64 = -0.367_879_441_171_442_33;

property! {
    #![config(settings("f64_lambert_w_backward").require("spans a decade or more", 50.0))]

    /// `W(x)·e^{W(x)} = x` across the whole domain, to within a few ulps of `W`'s resolution.
    ///
    /// The branch-point neighbourhood is **not** covered here and is not meant to be: over a
    /// range reaching `1e300` the generator put only 2.7% of cases within `0.05` of `−1/e`, which
    /// a coverage floor duly rejected. Hoping a broad sweep lands on the hardest argument is the
    /// mistake this suite exists to avoid, so that region gets its own generator in
    /// [`the_lambert_w_is_accurate_beside_its_branch_point`] and is exercised on every run.
    fn the_lambert_w_satisfies_its_defining_equation(
        x in generate::float::in_range(LAMBERT_BRANCH + 0.05, 1e300),
    ) {
        if !x.is_finite() || x < LAMBERT_BRANCH || x == 0.0 {
            return Ok(());
        }
        classify!("spans a decade or more", x.abs() > 1.0);

        let w = Real::lambert_w(x);
        if !w.is_finite() || w == 0.0 {
            return Ok(());
        }
        let back = w * w.exp();
        if !back.is_finite() {
            return Ok(());
        }
        // How far `x` moves for one ulp of `W`: the resolution of the answer.
        let up = {
            let v = f64::from_bits(w.to_bits() + 1);
            v * v.exp()
        };
        let down = {
            let v = f64::from_bits(w.to_bits() - 1);
            v * v.exp()
        };
        if !(up.is_finite() && down.is_finite()) {
            return Ok(());
        }
        let span = ((up - down) / x).abs();
        let backward = ((back - x) / x).abs();
        let allowed = LAMBERT_W_SPAN_SAFETY * span + 64.0 * f64::EPSILON;
        target(backward / allowed);
        check!(
            backward <= allowed,
            "lambert_w({x:e}) = {w:e} misses its defining equation by {backward:e}, over the \
             {allowed:e} its own resolution allows — one ulp of W spans {span:e} of x, and \
             W·e^W is {back:e}"
        );
    }
}

property! {
    #![config(settings("f64_lambert_w_branch").cases(800))]

    /// The same equation in the `0.05` beside `−1/e`, generated directly.
    ///
    /// `W` has a square-root singularity at the branch point, so `dW/dx` is infinite there and
    /// the Halley iteration is seeded from a series rather than from a logarithm. It is the
    /// hardest argument in the domain and the one a sweep over the full range cannot reach: the
    /// measured ratio of backward error to resolution peaks at exactly `2.00` here against
    /// `0.25` to `0.43` out in the tail.
    fn the_lambert_w_is_accurate_beside_its_branch_point(
        x in generate::float::in_range(LAMBERT_BRANCH, LAMBERT_BRANCH + 0.05),
    ) {
        if !x.is_finite() || x < LAMBERT_BRANCH {
            return Ok(());
        }
        let w = Real::lambert_w(x);
        if !w.is_finite() || w == 0.0 || x == 0.0 {
            return Ok(());
        }
        let back = w * w.exp();
        if !back.is_finite() {
            return Ok(());
        }
        let up = {
            let v = f64::from_bits(w.to_bits() + 1);
            v * v.exp()
        };
        let down = {
            let v = f64::from_bits(w.to_bits() - 1);
            v * v.exp()
        };
        if !(up.is_finite() && down.is_finite()) {
            return Ok(());
        }
        let span = ((up - down) / x).abs();
        let backward = ((back - x) / x).abs();
        let allowed = LAMBERT_W_SPAN_SAFETY * span + 64.0 * f64::EPSILON;
        target(backward / allowed);
        check!(
            backward <= allowed,
            "lambert_w({x:e}) = {w:e} misses its defining equation by {backward:e} beside the \
             branch point, over the {allowed:e} its resolution allows"
        );
    }
}
