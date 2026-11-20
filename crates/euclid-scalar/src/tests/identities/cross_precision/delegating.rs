//! Bit-exact checks for delegates, the eight-way agreement check and the mathematical constants.
//!
//! See the [parent module](super) for why this file is split from [`super::parameterised`].

use crate::Real;

// --- cross-precision agreement ------------------------------------------

#[test]
fn f32_and_f64_agree_to_single_precision() {
    // The same formula on both carriers must land within f32's own resolution. This is the
    // check that catches a divergent f32 code path rather than a mere precision difference.
    for i in 1..60 {
        let x64 = f64::from(i) * 0.1;
        let x32 = x64 as f32;
        let pairs: [(f64, f32); 8] = [
            (Real::erf(x64), Real::erf(x32)),
            (Real::erfc(x64), Real::erfc(x32)),
            (Real::ln_gamma(x64), Real::ln_gamma(x32)),
            (Real::digamma(x64), Real::digamma(x32)),
            (Real::trigamma(x64), Real::trigamma(x32)),
            (Real::sigmoid(&x64), Real::sigmoid(&x32)),
            (Real::ln_1p_exp(&x64), Real::ln_1p_exp(&x32)),
            (Real::bessel_i0e(x64), Real::bessel_i0e(x32)),
        ];
        for (idx, (a, b)) in pairs.iter().enumerate() {
            // Mixed criterion: `ln_gamma(1) = 0`, where a relative comparison between two
            // different roundings of zero is meaningless. f32 carries ~1.2e-7 resolution, so
            // the absolute floor has to accommodate it.
            let err = (f64::from(*b) - *a).abs();
            let bound = 1e-5 * a.abs().max(1.0);
            assert!(
                err <= bound,
                "f32/f64 diverged at x={x64}, fn index {idx}: {a} vs {b} \
                 (abs err {err:.3e} > {bound:.3e})"
            );
        }
    }
}

// ===== The f32 kernels against their f64 counterparts =====================
//
// # Why the test above is not enough
//
// `f32_and_f64_agree_to_single_precision` compares eight functions at a relative tolerance of
// `1e-5`. f32 carries about `1.2e-7` of resolution, so that bound is roughly eighty times
// looser than the format — wide enough that a mutated polynomial coefficient can sit inside it
// unnoticed. Mutation testing measured the consequence: around 480 surviving mutants in the f32
// tree, with `kernels/f32/quantile.rs` (118), `impls/real_f32.rs` (117),
// `kernels/f32/gamma.rs` (91), `kernels/f32/polygamma.rs` (77) and `kernels/f32/erf.rs` (74) the
// worst. `<impl Real for f32>::asin` could return `0.0` for every input.
//
// `euclid_stability::relation::precision_agrees` is the instrument built for exactly this, and
// it was called **nowhere in this crate**. Its docstring describes the risk precisely: an f32
// kernel "is written second, exercised less, and its constants are usually the f64 coefficients
// rounded — which is the right starting point and not always the right answer, because a
// polynomial fitted for f64 can be the wrong polynomial at f32".
//
// What it adds is the right bound. The wide result *rounded to f32* is the best the narrow
// kernel could possibly return, so the comparison is in f32 ulps against that — rather than a
// relative tolerance someone picked.
//
// # These are budgets, not ideals
//
// The numbers below are measured, in the same spirit as the monotonicity counts in
// `tests/stability.rs`: "the honest treatment is to measure and pin rather than to assert an
// ideal nobody promised". Today's behaviour passes; a kernel that drifts fails.
//
// Three things the measurement shows that are worth reading off it:
//
//   - **Nothing is bit-exact any more, and that is the end state rather than a regression.**
//     This bullet used to list eight kernels agreeing to zero ulp, each because it computed in
//     `f64` and rounded once. The whole `f32` surface is native now, so the test that pinned
//     those zeros has no members left and the list below is the entire picture. A budget of zero
//     across four hundred
//     arguments is not luck — these compute in f64 and round once. Pinning zero keeps that
//     true, and would catch someone "optimising" one into a genuine f32 path.
//
//     **Read that last sentence as a notification, not a prohibition.** Moving these to native
//     `f32` is the intended direction — an `f32` surface exists to be usable where `f64` is not,
//     so a kernel that reaches `f64` for its accuracy defeats its own purpose. When one is
//     ported, its entry moves out of this test and into the measured list below, with the new
//     budget derived rather than guessed. `erfc_inv` went that way first, from `0` here to `8`
//     there, and the single property that had been relying on the narrowing to assert *correct
//     rounding* was re-sited at the same time — see
//     `tests/properties/identities/distribution_extremes.rs`. A port that cannot state what it
//     gave up has not finished. `exp_int_e1` and `exp_int_ei` followed, to `13` and `49`, and
//     the `K` family to `14`. `J` followed, and is the one port so far whose budget here is set
//     by a function property rather than a method: `J` has infinitely many zeros, so an ulp
//     count against it collapses at every one and the number below says more about the metric
//     than the kernel. Against its envelope `√(2/πx)` — the scale a consumer's absolute error
//     actually has — the native `J` holds to `5.4e-7`, about four times the format's resolution.
//     The two numbers below land at `x = 8.65` and `x = 16.45`, which are `J₀`'s zero at `8.6537`
//     and `J₁`'s at `16.4706`. That is the metric reporting the zero, not the kernel. `Y₀`'s
//     `587` lands at `x = 7.08`, next to its zero at `7.0866`, for the same reason. Against
//     their envelopes the native `J` and `Y` hold to `5.4e-7` and `2.9e-7`. `airy_ai`'s `140`
//     and `airy_bi`'s `315` are the same story on the negative axis, where both oscillate
//     through zeros — against their envelope the native Airy holds to `7.0e-7`, and its
//     Wronskian `Ai·Bi′ − Ai′·Bi = 1/π` to `7.8e-7`.
//   - **The separately-implemented kernels are the loose ones**, and they are exactly the files
//     the survivors concentrated in: `gamma` 35, `bessel_i0` 48, `bessel_i1` 50.
//   - **A loose budget here has twice turned out to be a real defect**, so these numbers are
//     worth re-deriving rather than accepting. `erfc` and `erfcx` sat at **69** and **62** until
//     the `f32` `erf` module's continued-fraction threshold was found to have been *inherited*
//     from the `f64` peer instead of derived at `f32`'s epsilon; they are now **7** and **7**.
//     `ln_gamma` sat at **1514** and `digamma` at **205**, explained away here as the metric's
//     fault for collapsing near a zero — the zero is real, but it was hiding a native Lanczos
//     log form losing 20 ulps to cancellation wherever `lnΓ` is small. With the zeros factored
//     out instead of subtracted, they are **72** and **31**, and `trigamma` went 3 to 1.
//     The lesson is the one the file already states — measure and pin — plus: when a pinned
//     number is orders above its neighbours, suspect the kernel before the metric.
//   - **A port to native `f32` need not widen these numbers, and twice has narrowed them.**
//     `digamma` went 31 → **3** and `ln_gamma` 1514 → **72** on being ported, both *better* than
//     the `f64` kernel they used to narrow. That is not a paradox: this metric hands the wide
//     kernel `x` and the narrow one `x as f32`, so a narrowing kernel still pays for the
//     argument's rounding, and near a zero or a pole that term is the whole budget. A native
//     kernel that reproduces the zero by construction — `ψ(1+a) = (a − a₀)h(a)`,
//     `lnΓ(1+a) = a(a−1)g(a)` — beats it, because the factored form is *more* accurate at
//     exactly the arguments the metric is most sensitive at.
//
//     Where the port does cost something, say so: `erfc_inv` went 0 → **8** and `trigamma`
//     1 → **3**, and `erfc_inv` additionally gave up the correctly-rounded guarantee the
//     narrowing provided. At a common argument the native `ψ` and `ψ₁` hold to 4.5 and 2.5 times
//     the format's resolution over `[−9.9, −0.1]`.

/// Arguments swept per kernel. Fixed, because the budgets below are the maximum gap observed
/// over *this* sweep — changing it changes which arguments are visited and therefore the
/// measurement.
const SWEEP: usize = 400;

/// Check one kernel's f32 implementation against its f64 one across a domain.
macro_rules! agrees_to {
    ($name:literal, $lo:expr, $hi:expr, $ulps:expr, $wide:expr, $narrow:expr) => {{
        for i in 0..=SWEEP {
            #[allow(clippy::cast_precision_loss)]
            let x = $lo + ($hi - $lo) * (i as f64) / (SWEEP as f64);
            let outcome =
                euclid_stability::relation::precision_agrees($name, x, $wide, $narrow, $ulps);
            assert!(
                outcome.is_ok(),
                "{}",
                outcome.err().map(|f| f.message).unwrap_or_default()
            );
        }
    }};
}

// There is no longer a bit-exact test here, and the absence is deliberate.
//
// It pinned eight kernels at zero ulp — `bessel_j0`, `j1`, `k0`, `k1`, `y0`, `airy_ai`,
// `airy_bi`, `exp_int_ei`, `exp_int_e1`, `bessel_k0e`, `k1e`, `erfc_inv` — on the grounds that
// each computed in `f64` and rounded once, and said that pinning zero "would catch someone
// 'optimising' one into a genuine `f32` path". It caught every one of them, which is what it was
// for. The `f32` surface is now native throughout, so the test had no members left; an empty
// test body is a worse record of that than a comment, because it still reports as passing.
//
// What replaced it is the measured list below. The one property worth carrying forward from the
// old test is its reason for existing: a budget is only meaningful if someone re-derives it when
// the kernel changes, and a budget of zero was the cheapest possible tripwire for that. The
// measured budgets are not tripwires, so the discipline has to come from the practice described
// above instead.

/// The primitive-backed methods delegate to Rust's f32 implementations.
#[test]
fn the_f32_real_methods_match_the_native_primitives() {
    // Rust 1.95's native asinh(-7.88_f32) differs by one ulp from rounding f64 asinh.
    // Compare against the native delegate to check dispatch on every supported toolchain.
    macro_rules! matches_native {
        ($name:ident, $lo:expr, $hi:expr, $call:expr) => {{
            for i in 0..=SWEEP {
                let x = ($lo + ($hi - $lo) * (i as f64) / (SWEEP as f64)) as f32;
                let actual: f32 = ($call)(x);
                assert_eq!(
                    actual.to_bits(),
                    f32::$name(x).to_bits(),
                    "{}({x})",
                    stringify!($name)
                );
            }
        }};
    }
    matches_native!(asin, -0.99, 0.99, Real::asin);
    matches_native!(acos, -0.99, 0.99, Real::acos);
    matches_native!(atan, -8.0, 8.0, Real::atan);
    matches_native!(asinh, -8.0, 8.0, Real::asinh);
    matches_native!(acosh, 1.01, 10.0, Real::acosh);
    matches_native!(sinh, -5.0, 5.0, Real::sinh);
    matches_native!(cosh, -5.0, 5.0, Real::cosh);
    matches_native!(tanh, -5.0, 5.0, Real::tanh);
    matches_native!(cbrt, -20.0, 20.0, Real::cbrt);
    matches_native!(exp2, -10.0, 10.0, Real::exp2);
    matches_native!(exp_m1, -2.0, 2.0, |x: f32| Real::exp_m1(&x));
}

/// The mathematical constants must agree to the last bit of `f32`.
///
/// These are the simplest mutation target in the crate — a constant replaced by a default — and
/// nothing was checking them. There is no accuracy argument available here: the f32 value is the
/// f64 value correctly rounded, or it is wrong.
/// **Every constant on the trait, not a selection.** The list below used to hold four of the
/// twenty-four, and `sqrt_tau` surviving a mutation to `Default::default()` is what that costs:
/// there is no reason for one lifted constant to be checked and its neighbour not, and a
/// partial list gives no signal about which half it is in. A constant added to [`Real`] belongs
/// here the same day.
#[test]
fn the_constants_agree_across_precisions() {
    #[allow(clippy::cast_possible_truncation)]
    let narrowed = |wide: f64| wide as f32;

    // Each entry pairs the two carriers' accessors so the assertion below reads once.
    macro_rules! agree {
        ($($name:ident),* $(,)?) => {{
            $(assert_eq!(
                <f32 as Real>::$name(),
                narrowed(<f64 as Real>::$name()),
                concat!("the lifted constant `", stringify!($name), "`"),
            );)*
        }};
    }

    // The circle and its subdivisions.
    agree!(pi, tau, frac_pi_2, frac_pi_4, sqrt_pi, sqrt_tau);
    // Roots and their reciprocals.
    agree!(sqrt_2, frac_1_sqrt_2);
    // The logarithmic family.
    agree!(ln_2, ln_pi, ln_tau, half_ln_tau, ln_10);
    // The exponential family and the log bases.
    agree!(e, log2_e, log10_e, log2_10, log10_2);
    // The named analytic constants.
    agree!(euler_gamma, golden_ratio, apery, catalan, omega);
    // The small rationals, which are exact at both widths and so are the strictest of all.
    agree!(two, half, neg_one);
    // The degree constants, which exist so a formula can lift them without a literal.
    agree!(
        full_turn_degrees,
        half_turn_degrees,
        quarter_turn_degrees,
        three_half_turn_degrees,
    );

    // The non-finite three, which `narrowed` would compare incorrectly for NaN.
    assert_eq!(<f32 as Real>::infinity(), f32::INFINITY, "infinity");
    assert_eq!(
        <f32 as Real>::neg_infinity(),
        f32::NEG_INFINITY,
        "−infinity"
    );
    assert!(<f32 as Real>::nan().is_nan(), "NaN");
    assert!(<f64 as Real>::nan().is_nan(), "NaN at f64");
}

/// The two **derived** constants, which cannot make the bit-exact claim above — and the
/// distinction is worth stating rather than discovering.
///
/// Every constant in the test above is *lifted*: the body is `Self::scalar(literal)`, so the f32
/// value is the f64 value correctly rounded by construction and anything else is a typo.
/// `radians_per_degree` and `degrees_per_radian` are **computed**, as `pi / 180` and `180 / pi`,
/// so each carrier divides using *its own* `pi` and the results are not related by rounding.
///
/// Extending the list above to cover everything caught this: `degrees_per_radian` at f32 is
/// `57.295_776`, while the f64 value narrowed is `57.295_780` — some thirty-two f32 ulps apart.
/// That is not a defect. `pi` at f32 is already rounded, and putting it in a *denominator*
/// carries that error into the quotient, where rounding the f64 quotient instead keeps it. Its
/// sibling `radians_per_degree` happens to agree bit-for-bit, which is exactly the kind of
/// coincidence that would make a selective list look fine.
///
/// So these two are checked against the true value at each carrier's own resolution, which is
/// the only claim either can support.
#[test]
fn the_derived_angle_constants_match_their_definitions() {
    // 180/π and π/180 from an independent source, written to the digits `f64` can hold —
    // the true values continue `...320_877` and `...295_769`.
    const DEGREES_PER_RADIAN: f64 = 57.295_779_513_082_32;
    const RADIANS_PER_DEGREE: f64 = 0.017_453_292_519_943_295;

    let close = |got: f64, want: f64, tol: f64, what: &str| {
        assert!(
            (got / want - 1.0).abs() <= tol,
            "{what}: got {got}, want {want} (relative gap {:.3e} exceeds {tol:.0e})",
            (got / want - 1.0).abs(),
        );
    };

    // f64: both land within an ulp of the true value.
    close(
        <f64 as Real>::degrees_per_radian(),
        DEGREES_PER_RADIAN,
        1e-15,
        "f64 degrees_per_radian",
    );
    close(
        <f64 as Real>::radians_per_degree(),
        RADIANS_PER_DEGREE,
        1e-15,
        "f64 radians_per_degree",
    );
    // f32: measured at 5.6e-8 and 7.8e-9 respectively, both inside one f32 ulp of the truth.
    close(
        f64::from(<f32 as Real>::degrees_per_radian()),
        DEGREES_PER_RADIAN,
        2e-7,
        "f32 degrees_per_radian",
    );
    close(
        f64::from(<f32 as Real>::radians_per_degree()),
        RADIANS_PER_DEGREE,
        2e-7,
        "f32 radians_per_degree",
    );

    // And they are reciprocals, at each carrier's own precision.
    for (product, what) in [
        (
            <f64 as Real>::degrees_per_radian() * <f64 as Real>::radians_per_degree(),
            "f64",
        ),
        (
            f64::from(<f32 as Real>::degrees_per_radian() * <f32 as Real>::radians_per_degree()),
            "f32",
        ),
    ] {
        close(
            product,
            1.0,
            2e-7,
            &format!("{what}: the two are reciprocals"),
        );
    }
}

/// The f32 kernels with their own implementation, each at its measured budget.
#[test]
fn the_independent_f32_kernels_stay_inside_their_budgets() {
    agrees_to!("erf", -4.0, 4.0, 8, Real::erf, Real::erf);
    agrees_to!("erfc", -4.0, 4.0, 7, Real::erfc, Real::erfc);
    agrees_to!("erfcx", 0.01, 10.0, 7, Real::erfcx, Real::erfcx);
    agrees_to!("erfc_inv", 0.0, 2.0, 8, Real::erfc_inv, Real::erfc_inv);
    agrees_to!(
        "exp_int_e1",
        0.1,
        10.0,
        13,
        Real::exp_int_e1,
        Real::exp_int_e1
    );
    // `Ei` has a zero at `x ≈ 0.3725`, inside this sweep, so the budget is dominated by the
    // metric collapsing there rather than by the kernel — the same effect noted for
    // `ln_gamma`, `digamma` and `atanh` below. Measured 49; the native `Ei` holds to 4 ulp
    // against the `f64` kernel away from the zero.
    agrees_to!(
        "exp_int_ei",
        0.1,
        10.0,
        49,
        Real::exp_int_ei,
        Real::exp_int_ei
    );
    agrees_to!("erf_inv", -0.99, 0.99, 15, Real::erf_inv, Real::erf_inv);
    agrees_to!("gamma", 0.1, 8.0, 35, Real::gamma, Real::gamma);
    agrees_to!("trigamma", 0.1, 20.0, 3, Real::trigamma, Real::trigamma);
    agrees_to!("bessel_i0", 0.0, 15.0, 48, Real::bessel_i0, Real::bessel_i0);
    agrees_to!(
        "bessel_i0e",
        0.0,
        30.0,
        44,
        Real::bessel_i0e,
        Real::bessel_i0e
    );
    agrees_to!("bessel_i1", 0.0, 15.0, 50, Real::bessel_i1, Real::bessel_i1);
    agrees_to!("exprel", -2.0, 2.0, 1, Real::exprel, Real::exprel);
    agrees_to!("cos_m1", -3.0, 3.0, 2, Real::cos_m1, Real::cos_m1);
    agrees_to!(
        "sigmoid",
        -10.0,
        10.0,
        1,
        |v: f64| Real::sigmoid(&v),
        |v: f32| Real::sigmoid(&v)
    );
    agrees_to!(
        "ln_1p_exp",
        -10.0,
        10.0,
        1,
        |v: f64| Real::ln_1p_exp(&v),
        |v: f32| Real::ln_1p_exp(&v)
    );

    // Measured in a second pass over the surface the first one missed.
    agrees_to!("sinc", -8.0, 8.0, 1, Real::sinc, Real::sinc);
    agrees_to!(
        "ln_1m_exp",
        -8.0,
        -0.01,
        1,
        |v: f64| Real::ln_1m_exp(&v),
        |v: f32| Real::ln_1m_exp(&v)
    );
    agrees_to!(
        "bessel_i1_over_i0",
        0.1,
        20.0,
        7,
        Real::bessel_i1_over_i0,
        Real::bessel_i1_over_i0
    );
    agrees_to!(
        "bessel_i1e",
        0.0,
        30.0,
        43,
        Real::bessel_i1e,
        Real::bessel_i1e
    );
}

/// The kernels whose budget is dominated by a zero inside the range, not by the kernel.
///
/// See the note at the top of this section: an ulp count against a function that passes through
/// zero measures the metric's collapse rather than the implementation's error. Pinned anyway,
/// because a *rise* here is still a regression worth failing on.
///
/// **But the collapse is not a licence to stop looking.** `ln_gamma` was pinned at 1514 and
/// `digamma` at 205 on exactly this reasoning, and both were also carrying a genuine
/// cancellation defect — the two causes are not exclusive, and the metric's noise was large
/// enough to hide the kernel's. At 72 and 31 the remaining width really is the zero.
#[test]
fn the_f32_kernels_with_a_root_in_range_are_pinned_loosely() {
    agrees_to!("ln_gamma", 0.1, 20.0, 72, Real::ln_gamma, Real::ln_gamma);
    agrees_to!("digamma", 0.1, 20.0, 3, Real::digamma, Real::digamma);
    // `atanh(0) = 0`, so the same collapse applies — 17 ulp here is the metric, not the kernel.
    agrees_to!("atanh", -0.99, 0.99, 17, Real::atanh, Real::atanh);
}
