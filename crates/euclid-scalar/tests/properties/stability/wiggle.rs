//! Ulp-level monotonicity, and measured budgets for the kernels that lack it.

use super::*;

// ===== Wiggle: the kernels that are ulp-monotone ==========================

/// These are monotone to the last bit over 50,000 consecutive inputs, which is a real quality
/// property worth protecting. An inverse or a root-find over any of them can safely iterate to
/// the format's resolution.
#[test]
fn the_ulp_monotone_kernels_stay_monotone() {
    let clean: Vec<MonotoneCase> = vec![
        ("erf", Real::erf, 0.5, true),
        ("erfc", Real::erfc, 0.5, false),
        ("erfc", Real::erfc, 2.0, false),
        ("digamma", Real::digamma, 3.0, true),
        ("tanh", f64::tanh, 1.0, true),
        ("exp", f64::exp, 1.0, true),
        ("ln", f64::ln, 3.0, true),
        // Measured clean below, and previously unscanned. The original list was six distinct
        // kernels out of seventy-five on the trait, almost all of them `erf` or gamma.
        ("ndtr", Real::ndtr, 0.5, true),
        ("ndtr", Real::ndtr, -3.0, true),
        ("log_ndtr", Real::log_ndtr, -3.0, true),
        ("log_ndtr", Real::log_ndtr, 0.5, true),
        ("trigamma", Real::trigamma, 1.0, false),
        ("trigamma", Real::trigamma, 5.0, false),
        ("lambert_w", Real::lambert_w, 1.0, true),
        ("zeta", Real::zeta, 2.0, false),
        ("zeta", Real::zeta, 5.0, false),
        ("exp_int_ei", Real::exp_int_ei, 1.0, true),
        ("exp_int_ei", Real::exp_int_ei, 40.0, true),
        ("exprel", Real::exprel, 1.0, true),
        ("cbrt", Real::cbrt, 2.0, true),
        ("asinh", Real::asinh, 1.0, true),
        ("acosh", Real::acosh, 2.0, true),
        ("atanh", Real::atanh, 0.5, true),
        // Away from their crossovers these are clean; nearer in they wiggle, and those cases are
        // budgeted in `the_newly_scanned_kernels_stay_within_their_wiggle_budgets`.
        ("bessel_k0", Real::bessel_k0, 5.0, false),
        ("bessel_k0e", Real::bessel_k0e, 5.0, false),
        ("bessel_i0", Real::bessel_i0, 5.0, true),
        ("airy_ai", Real::airy_ai, 5.0, false),
        ("airy_bi", Real::airy_bi, 1.0, true),
        ("airy_bi", Real::airy_bi, 5.0, true),
        ("exp_int_e1", Real::exp_int_e1, 10.0, false),
        // `E₁` at `1.0` is **no longer here**, and the history is worth keeping because it has
        // now gone both ways. It was budgeted with the worst inversion in the suite — `114` ulp
        // — when `1.0` was the first argument on the continued fraction; it became bit-monotone
        // when `E1_SERIES_MAX` rose to `2.0` and left it the *last* argument on the series; and
        // it is budgeted again now that the series has given `[7/8, 4]` to a Chebyshev fit, at
        // `1` ulp over `32` of `50000` pairs. See
        // `the_newly_scanned_kernels_stay_within_their_wiggle_budgets`.
        // The last fifteen kernels the scan had never reached. Thirteen start points among them
        // are clean to the last bit, which is more than the earlier rounds found — the
        // oscillatory families are *better* behaved here than the gamma ones, away from their
        // crossovers. Each direction below is measured rather than reasoned about: the scan
        // window is about `50,000` ulps wide, which at `x = 1` is `1.1e-11`, so every start sits
        // deep inside one monotone arc and the sign is not in doubt.
        ("airy_ai_prime", Real::airy_ai_prime, 5.0, true),
        ("airy_bi_prime", Real::airy_bi_prime, 2.0, true),
        ("airy_bi_prime", Real::airy_bi_prime, 5.0, true),
        ("bessel_k1", Real::bessel_k1, 5.0, false),
        ("cos_m1", Real::cos_m1, 0.1, false),
        ("cos_m1", Real::cos_m1, 1.0, false),
        ("bessel_j0", Real::bessel_j0, 1.0, false),
        ("bessel_j1", Real::bessel_j1, 1.0, true),
        ("bessel_y0", Real::bessel_y0, 1.0, true),
        ("bessel_y0", Real::bessel_y0, 5.0, false),
        ("bessel_y1", Real::bessel_y1, 5.0, false),
        ("cos_int", Real::cos_int, 1.0, true),
        ("sin_int", Real::sin_int, 1.0, true),
    ];

    for (name, f, start, increasing) in &clean {
        let inversion = scan_monotone(|x| if *increasing { f(x) } else { -f(x) }, *start, STEPS);
        assert!(
            inversion.is_none(),
            "{name} near {start} lost ulp-level monotonicity: {inversion:?}"
        );
    }
}

// ===== Wiggle: the measured budgets ======================================

/// `erf` is very nearly ulp-monotone — four inverting pairs in fifty thousand, never by more
/// than one ulp. Fine for anything; recorded so a regression is visible.
#[test]
fn erf_wiggle_stays_within_its_budget() {
    assert_budget("erf", Real::erf, 2.0, true, 8, 1.0);
}

/// `sinc` inverts on about 6% of pairs by a single ulp. Its derivative near `0.5` is small, so
/// the true step between adjacent inputs is comparable to the kernel's own rounding — which is
/// the mechanism behind every entry in this section.
#[test]
fn sinc_wiggle_stays_within_its_budget() {
    assert_budget("sinc", Real::sinc, 0.5, false, 3_300, 1.0);
}

/// `ln_gamma` near 3 is the worst case found: a third of adjacent pairs step backwards, by up
/// to 24 ulps. Consistent with the documented `1e-14` relative accuracy, and the reason not to
/// bisect on it past about twelve digits.
#[test]
fn ln_gamma_wiggle_stays_within_its_budget() {
    assert_budget("ln_gamma", Real::ln_gamma, 3.0, true, 16_500, 32.0);
    assert_budget("ln_gamma", Real::ln_gamma, 10.0, true, 3_300, 4.0);
}

/// `gamma` near 3 inverts on about a quarter of pairs, by up to 9 ulps.
#[test]
fn gamma_wiggle_stays_within_its_budget() {
    assert_budget("gamma", Real::gamma, 3.0, true, 12_500, 12.0);
}

/// `erfcx` near 1 inverts on about a third of all adjacent pairs, by up to 21 ulps — the highest
/// inversion *rate* measured, though not the largest single inversion.
///
/// # Re-recorded when `erf` stopped routing through the incomplete gamma
///
/// The worst inversion rose from `15` to `21` ulp while the *rate* fell from about `25000` pairs
/// to `19372`, and accuracy over the same scanned run improved from `51` to `29` ulp. All three
/// move together for one reason: `erfcx` here is `e^{x²}(1 − erf(x))`, which magnifies `erf`'s
/// error by `1/erfc(1) ≈ 6.4`, and `erf`'s error changed *in kind*. The old incomplete-gamma
/// route carried a large but **smooth** bias — `exp` of a cancelling sum, so the error was a
/// slowly varying function of `x`. The series carries a small but **random** one.
///
/// A smooth bias barely inverts adjacent pairs however big it is; last-bit noise inverts them
/// constantly however small it is. So a kernel can get materially more accurate and measure
/// *less* monotone, and these two numbers moving in opposite directions is that happening rather
/// than anything going wrong. Budgets re-measured, not relaxed to fit.
#[test]
fn erfcx_wiggle_stays_within_its_budget() {
    assert_budget("erfcx", Real::erfcx, 1.0, false, 25_000, 26.0);
}

/// Wiggle budgets for the kernels the widened scan reached for the first time.
///
/// # `E₁` was the outlier, and moving its crossover is what fixed it
///
/// Every other kernel measured here inverts by at most `8` ulp. `E₁` inverted by **114**, and the
/// shape of it located the cause precisely:
///
/// ```text
/// x               0.1   0.5   0.9   0.999    1.0   1.001    1.1    2.0     10
/// worst inversion   1     0     0       4    102     114     76     51      0
/// ```
///
/// `E1_SERIES_MAX` was `1.0`: below it `E₁` sums a power series, above it runs Lentz's continued
/// fraction. The fraction converges slowest exactly where it takes over, so its truncation noise
/// was largest just above `1`. `Ei` shares the module and is clean at `0.5` and `40`, so it was
/// the fraction rather than the family.
///
/// Measured against a 90-digit reference, the series is the *better* route all the way to about
/// `x = 2` — `0`–`4` ulp against the fraction's `12`–`22` — so the threshold was a decade too low
/// and every argument in `[1, 2]` took the worse of the two. It is now `2.0`, and the table reads:
///
/// ```text
/// x               0.999   1.0   1.001   1.1    1.5    1.9    2.0    2.1     10
/// worst inversion     4     0      4      4      8     64     51     34      0
/// ```
///
/// `E₁` is **exactly monotone at `1.0`** now, and the peak has moved to `1.9` — the top of the
/// series, where its own alternating cancellation is largest — and fallen from `114` to `64`.
///
/// The seam scan passed throughout, which is not a contradiction: the two routes *agree* at the
/// join to well within the seam tolerance, and still disagreed enough to put `114` ulp of
/// non-monotonicity beside it. A seam test and a wiggle test ask different questions.
///
/// Practical consequence, revised: a root-find on `E₁` is now safe below `1.5` and still has to
/// stop at about `64` ulp between there and `3`.
///
/// # The three inverses wiggle too, and that is expected
///
/// `erf_inv`, `erfc_inv` and `ndtri` all invert by a few ulp. That is the same effect measured
/// from the other side in `the_normal_quantile_inverts_its_own_cdf`: several adjacent arguments
/// map to the same probability, so the inverse has no single right answer among them and steps
/// between them as the argument advances.
#[test]
fn the_newly_scanned_kernels_stay_within_their_wiggle_budgets() {
    // (kernel, start, increasing, inversion budget, ulp budget) — each measured, then rounded up
    // for headroom so the budget catches a regression rather than noise.
    // `1.0` moved to the bit-monotone list when the crossover rose; these are the series' own
    // residual, measured after the change — `64` ulp at `1.9` and `51` at `2.0`.
    assert_budget("exp_int_e1", Real::exp_int_e1, 1.5, false, 26_000, 16.0);
    assert_budget("exp_int_e1", Real::exp_int_e1, 1.9, false, 16_000, 128.0);
    assert_budget("exp_int_e1", Real::exp_int_e1, 2.0, false, 8_000, 102.0);
    assert_budget("ln_bessel_i0", Real::ln_bessel_i0, 1.0, true, 18_000, 16.0);
    assert_budget("bessel_i1", Real::bessel_i1, 1.0, true, 28_000, 4.0);
    assert_budget("bessel_k1e", Real::bessel_k1e, 1.0, false, 22_000, 10.0);
    assert_budget("bessel_i0e", Real::bessel_i0e, 5.0, false, 18_000, 12.0);
    assert_budget("airy_ai_e", Real::airy_ai_e, 2.0, false, 20_000, 6.0);
    assert_budget("airy_bi_e", Real::airy_bi_e, 2.0, false, 16_000, 8.0);
    // Re-measured when `erf` left the incomplete gamma: inversions `1500 -> 2251` while the
    // worst inversion fell `8 -> 5` and accuracy over the same run improved `12 -> 7` ulp. See
    // `erfcx_wiggle_stays_within_its_budget` for why a more accurate kernel inverts more often.
    assert_budget("erf_inv", Real::erf_inv, 0.5, true, 2_600, 8.0);
    assert_budget("erfc_inv", Real::erfc_inv, 0.5, false, 2_000, 10.0);
    assert_budget("ndtri", Real::ndtri, 0.3, true, 4_500, 6.0);
    // `E₁` at `1.0`, which the fit on `[7/8, 4]` took off the bit-monotone list. Measured `32`
    // of `50000` pairs inverting by `1` ulp, against the `17`-to-`43` ulp of *accuracy* the
    // series carried over that window — an approximation is not bit-monotone where the series it
    // replaced happened to be, and a smooth route's last bits wobble wherever its error does.
    // The window's lower edge is already an eighth below `1` for the related reason: see
    // `E1_CHEBYSHEV_MIN`, which explains why a boundary should not sit on a scanned argument.
    assert_budget("exp_int_e1", Real::exp_int_e1, 1.0, false, 200, 4.0);
}

/// Wiggle budgets for the last fifteen kernels, and the two mechanisms they expose.
///
/// These close the scan: every kernel [`super::range`] covers now has a monotonicity figure.
/// Thirteen of their start points are bit-clean and sit in the list above; the twenty below
/// wiggle, and two of them do so for reasons worth separating from the rest.
///
/// # `Ci` peaks *at* its crossover, which is a different shape from `E₁`
///
/// `TRIG_SERIES_MAX` is `4.0`. Worst inversion either side of it:
///
/// ```text
/// x                 1    2    3   3.9   3.999    4    4.001   4.1    5    8    20   100
/// worst inversion   0    0   32    64     32    68     51     47    32   27    4     0
/// ```
///
/// Compare the `E₁` table in [`the_newly_scanned_kernels_stay_within_their_wiggle_budgets`],
/// which is `0` below its crossover and `102` immediately above: **one-sided**. `Ci` is elevated
/// on *both* sides and peaks on the join. So this is not "the continued fraction is noisy where it
/// takes over" — it is the series stretched at the top of its range meeting the fraction stretched
/// at the bottom of its, with the worst of both at the handover. A seam test cannot see it, for
/// the reason given in the `E₁` note: the two routes agree at the join far inside the seam
/// tolerance and still leave `68` ulp of non-monotonicity around it.
///
/// # And `Ci` is thirteen times `Si` because of its *value*, not its route
///
/// `Si` shares the module, the crossover and the argument reduction, and peaks at `5` ulp against
/// `Ci`'s `68`. The ratio is not a quality difference: `Si(4) = 1.758` and `Ci(4) = −0.141`, a
/// factor of `12.5`, against a measured ulp ratio of `13.6`. Their *absolute* errors are
/// comparable and `Ci` simply has a smaller value for an ulp to be measured against — `Ci` has a
/// zero at `3.384`, close enough to inflate the count without being near enough to dominate it.
/// That is the same correction [`super::cross_precision`] makes with its envelopes.
///
/// # `rgamma` at large argument is `exp`'s conditioning, and its shape says so
///
/// `1/Γ` is the one kernel here whose wiggle *grows* without bound:
///
/// ```text
/// x                 0.1   0.5   1   1.46   2    3    5   10    50
/// worst inversion    14     9   8     9   32   24    6   24   174
/// ```
///
/// At `x = 50` it inverts by `174` ulp on only **97** of fifty thousand pairs — a hundredfold
/// fewer inversions than its neighbours, each a hundred times larger. Both halves follow from the
/// route: `1/Γ(x)` for large `x` is `e^{−lnΓ(x)}`, and `exp`'s condition number is its own
/// argument, so one ulp in `lnΓ` becomes `|lnΓ(x)|` ulps here. `|lnΓ(50)| = 144.57` against a
/// measured `174`, which is the agreement that identifies the mechanism. The low inversion count
/// is the other side of it: `1/Γ` falls steeply there, so almost every step is decisively
/// downward and only the rare one lands backwards — but when it does, it lands far.
///
/// The practical reading, and the reason this is a budget rather than a bound: a root-find on
/// `rgamma` loses roughly `log₂|lnΓ(x)|` bits, so about seven of them at `x = 50` and more
/// further out. Nothing is wrong with the kernel; the amplification is in the composition.
#[test]
fn the_remaining_unscanned_kernels_stay_within_their_wiggle_budgets() {
    // (kernel, start, increasing, inversion budget, ulp budget) — measured, then roughly doubled,
    // the same headroom convention as the section above.
    assert_budget("airy_ai_prime", Real::airy_ai_prime, 2.0, true, 3_300, 6.0);
    assert_budget(
        "airy_ai_e_prime",
        Real::airy_ai_e_prime,
        2.0,
        false,
        18_000,
        14.0,
    );
    assert_budget(
        "airy_ai_e_prime",
        Real::airy_ai_e_prime,
        20.0,
        false,
        14_000,
        8.0,
    );
    assert_budget(
        "airy_bi_e_prime",
        Real::airy_bi_e_prime,
        2.0,
        true,
        20_000,
        10.0,
    );
    assert_budget(
        "airy_bi_e_prime",
        Real::airy_bi_e_prime,
        20.0,
        true,
        14_000,
        4.0,
    );
    assert_budget("bessel_i1e", Real::bessel_i1e, 1.0, true, 28_000, 8.0);
    assert_budget("bessel_i1e", Real::bessel_i1e, 5.0, false, 18_000, 12.0);
    assert_budget(
        "bessel_i1_over_i0",
        Real::bessel_i1_over_i0,
        1.0,
        true,
        600,
        2.0,
    );
    assert_budget(
        "bessel_i1_over_i0",
        Real::bessel_i1_over_i0,
        5.0,
        true,
        24_000,
        10.0,
    );
    // `2_800`, raised from `2_400` when `I₁` stopped recovering its unscaled value through
    // `exp(ln I₁ + x)`. `k1_series` is `1/x + ln(x/2)·I₁(x) − x·Σ/4`, a cancelling combination
    // in which `ln(x/2)` passes through zero at `x = 2`, so any change to `I₁`'s low bits moves
    // this count. It measured 2,514.
    //
    // Raised rather than investigated, because the accuracy went the other way: against a
    // reference built from the Wronskian `I₀K₁ + I₁K₀ = 1/x` with MPFR's `I₀`, `I₁` and `K₀`,
    // `K₁`'s worst relative error over `(0.05, 2]` *fell* from `1.653e-15` to `1.490e-15`. An
    // inversion count at this spacing is a count of ulp-level noise, and the noise moved.
    assert_budget("bessel_k1", Real::bessel_k1, 1.0, false, 2_800, 4.0);
    assert_budget("bessel_j0", Real::bessel_j0, 5.0, true, 40, 2.0);
    assert_budget("bessel_j1", Real::bessel_j1, 5.0, false, 4_000, 6.0);
    assert_budget("bessel_y1", Real::bessel_y1, 1.0, true, 40, 2.0);
    assert_budget("sin_int", Real::sin_int, 5.0, false, 3_500, 4.0);
    // `Ci` across its crossover, per the table above.
    assert_budget("cos_int", Real::cos_int, 4.0, false, 20_000, 136.0);
    assert_budget("cos_int", Real::cos_int, 8.0, false, 24_000, 54.0);
    // `1/Γ`, whose budget has to grow with `|lnΓ|` rather than sit at a constant.
    // `12_500`, raised from `10_000` when `rgamma` stopped forming `1/Γ` as `exp(−lnΓ(x))`.
    // Measured 11,823. The accuracy went the other way: against MPFR over a tight band at `0.5`
    // and `[0.1, 2.1]`, the worst relative error *fell* from `2.06e-15` to `1.59e-15`, and the
    // inversion count over that same sample is identical at 64. What moved is sub-ulp noise at
    // the 50,000-point spacing this scans, not monotonicity the caller could observe.
    assert_budget("rgamma", Real::rgamma, 0.5, true, 12_500, 18.0);
    assert_budget("rgamma", Real::rgamma, 2.0, false, 20_000, 64.0);
    assert_budget("rgamma", Real::rgamma, 3.0, false, 22_000, 48.0);
    assert_budget("rgamma", Real::rgamma, 50.0, false, 400, 348.0);
}
