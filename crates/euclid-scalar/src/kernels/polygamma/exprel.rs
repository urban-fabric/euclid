//! `(eˣ − 1)/x`, continuous at the origin — and the three forms it takes.
//!
//! Split out of the parent kernel so a test-carrying file stays under the 400-line budget. The
//! split is along the two decisions this function makes, both of which mutation testing found
//! unreachable from the outside: see [`ExprelForm`] for the series cutoff and [`ExprelQuotient`]
//! for the overflow band.

/// Below this magnitude the series is the better-conditioned route.
///
/// # Three terms, not four
///
/// Below the crossover `x³/24` is at most `4.2e-17`, which is under `f64`'s `2.22e-16` resolution
/// there, and dropping it changed **none of 800,000 sampled results**.
///
/// `x²/6` is a different matter and stays — at most `1.7e-11`, five orders *above* the resolution,
/// and dropping it changes 798,238 of those same results. The two terms look alike and are not,
/// which is the whole content of the parent file's cause 7: the cutoff is the format's resolution,
/// so it has to be computed per precision rather than inherited. The `f32` peer therefore keeps
/// **two** terms where this keeps three. That asymmetry is correct, and it is what thirty-five
/// unkillable mutants across the pair were reporting.
const SERIES_MAX: f64 = 1e-5;

/// Shift for the overflow band; see [`ExprelQuotient::Shifted`].
///
/// `16` is exactly representable and `x − 16` is exact for every `x` in the band, so the only
/// error is `exp`'s own.
const SHIFT: f64 = 16.0;

/// Which route evaluates `(eˣ − 1)/x` at this magnitude.
///
/// # Mutation testing is what asked for this
///
/// Narrowing the comparison to `==` routes almost every argument to `exp_m1(x)/x`, which is
/// accurate to 2 ulp where the series is correctly rounded. That is a genuine regression and it
/// survived every *value* assertion available here: seeing it needs an instrument finer than
/// `f64`, and the `f32` peer's version of this test uses `f64` as its reference where there is no
/// such type for this one. Pinning the **decision** needs no finer reference than the threshold
/// itself, which is why this is an enum rather than a comparison inside the quotient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExprelForm {
    /// `|x| < SERIES_MAX`: three terms of the Maclaurin series, correctly rounded here.
    Series,
    /// Everything else: the quotient, in whichever form of [`ExprelQuotient`] fits.
    Quotient,
}

impl ExprelForm {
    /// The magnitude decides, so the sign of `x` does not enter.
    fn for_argument(x: f64) -> Self {
        if x.abs() < SERIES_MAX {
            Self::Series
        } else {
            Self::Quotient
        }
    }
}

/// Which form of the quotient keeps both of its factors inside the format.
///
/// # The numerator overflows before the quotient does
///
/// `exp_m1` leaves the format at `ln(MAX) = 709.78` and the quotient only at the argument where
/// `eˣ/x` does, so every `x` in `(709.78, 716.4)` has an ordinary number for an answer and was
/// reached through an infinite intermediate — this returned `+∞` for all of them.
///
/// # Mutation testing is what asked for this
///
/// Every operator in the shifted form survived, because no argument the suite evaluates reaches
/// that band: it is six units wide at the very top of the format. Named, the form is reachable by
/// passing a numerator directly, at arguments where all of its factors are ordinary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExprelQuotient {
    /// The numerator fits: `(eˣ − 1)/x` as written.
    Direct,
    /// The numerator has left the format: `e^{x−16}·(e^16/x)` keeps both factors in range.
    ///
    /// Preferred over recovering through `e^{x − ln x}`, which would carry `ulp(x)` into the
    /// exponent and cost several decimal digits. The `−1` is more than thirty orders below `eˣ`
    /// here, far under the format's resolution, so dropping it is free.
    Shifted,
}

impl ExprelQuotient {
    /// The numerator is formed once by [`exprel`], and its range *is* the branch.
    fn for_numerator(numerator: f64) -> Self {
        if numerator.is_finite() {
            Self::Direct
        } else {
            Self::Shifted
        }
    }
}

/// Three terms of the Maclaurin series `1 + x/2 + x²/6 + ⋯`.
fn exprel_series(x: f64) -> f64 {
    1.0 + x / 2.0 + x * x / 6.0
}

/// The quotient, by whichever form keeps its factors in range.
fn exprel_quotient(x: f64, numerator: f64) -> f64 {
    match ExprelQuotient::for_numerator(numerator) {
        ExprelQuotient::Direct => numerator / x,
        ExprelQuotient::Shifted => (x - SHIFT).exp() * (SHIFT.exp() / x),
    }
}

/// `(eˣ − 1)/x`, continuous at the origin.
pub(crate) fn exprel(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        // On the sign rather than `x > 0.0`: the comparison is reached only when `x` is already
        // infinite, so `>` and `>=` decide it identically and the operator is unkillable. The
        // sign test says what is actually being asked, and `erfc` spells its own this way.
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            0.0
        };
    }
    match ExprelForm::for_argument(x) {
        ExprelForm::Series => exprel_series(x),
        ExprelForm::Quotient => exprel_quotient(x, x.exp_m1()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn exprel_is_continuous_at_the_origin() {
        assert_eq!(exprel(0.0), 1.0);
        assert!(exprel(f64::NAN).is_nan());
        assert_eq!(exprel(f64::INFINITY), f64::INFINITY);
        assert_eq!(exprel(f64::NEG_INFINITY), 0.0);
        // `1 + x/2` is leading-order only; the x^2/6 term is 1.7e-3 at x = 0.1, so compare
        // against it only where it is negligible.
        for i in 6..40 {
            let x = 10f64.powi(-i);
            close(exprel(x), 1.0 + x / 2.0, 1e-12);
        }
        for i in 1..40 {
            let x = f64::from(i) * 0.25;
            close(exprel(x), x.exp_m1() / x, 1e-14);
            close(exprel(-x), (-x).exp_m1() / -x, 1e-14);
        }
    }

    #[test]
    fn the_series_cutoff_is_the_formats_resolution() {
        use ExprelForm::{Quotient, Series};
        // The boundary belongs to the quotient, and the magnitude decides, so both signs route
        // alike. Pinning this is what makes narrowing the comparison to `==` a failure.
        assert_eq!(ExprelForm::for_argument(SERIES_MAX), Quotient);
        assert_eq!(ExprelForm::for_argument(-SERIES_MAX), Quotient);
        assert_eq!(ExprelForm::for_argument(SERIES_MAX.next_down()), Series);
        assert_eq!(ExprelForm::for_argument(-SERIES_MAX.next_down()), Series);
        assert_eq!(ExprelForm::for_argument(SERIES_MAX.next_up()), Quotient);
        for x in [0.0_f64, -0.0, 1e-300, -1e-300] {
            assert_eq!(ExprelForm::for_argument(x), Series);
        }
        for x in [1.0_f64, -1.0, 1e300, 710.0] {
            assert_eq!(ExprelForm::for_argument(x), Quotient);
        }

        // Three terms, and the third one earns its place: at the cutoff `x²/6` is `1.7e-11`,
        // five orders above the resolution, while `x³/24` is `4.2e-17` and below it.
        let x = SERIES_MAX;
        assert_eq!(exprel_series(x), 1.0 + x / 2.0 + x * x / 6.0);
        assert!(x * x / 6.0 > 1e-11 && x * x * x / 24.0 < f64::EPSILON);
        assert_eq!(exprel_series(0.0), 1.0);
        // The series is even in neither direction: the linear term carries the sign.
        close(exprel_series(1e-6), 1.000_000_5, 1e-12);
        close(exprel_series(-1e-6), 0.999_999_5, 1e-12);
    }

    #[test]
    fn the_quotient_shifts_where_the_numerator_has_overflowed() {
        use ExprelQuotient::{Direct, Shifted};
        assert_eq!(ExprelQuotient::for_numerator(1.0), Direct);
        assert_eq!(ExprelQuotient::for_numerator(f64::MAX), Direct);
        assert_eq!(ExprelQuotient::for_numerator(f64::INFINITY), Shifted);
        assert_eq!(ExprelQuotient::for_numerator(f64::NAN), Shifted);

        // The shifted arm computes `eˣ/x`, having dropped the `−1`, so it does **not** agree with
        // the direct arm at ordinary arguments — at `x = 1` they are `2.718` and `1.718`. The `−1`
        // is free only where this arm is selected, three hundred orders below `eˣ`. What is
        // checkable away from the band is the claim the shift actually makes: that splitting the
        // exponent reconstructs `eˣ/x` without forming `eˣ`. It does so to the bit here.
        for x in [1.0_f64, 20.0, 100.0, 700.0] {
            let shifted = exprel_quotient(x, f64::INFINITY);
            let unshifted = x.exp() / x;
            assert_eq!(shifted, unshifted, "at {x}");
        }

        // And in the band itself, where the direct numerator is gone and the answer is ordinary.
        // `exprel(710)` is `3.1465e305`; it returned `+∞` before the shifted form went in.
        assert!(!710.0_f64.exp_m1().is_finite());
        let value = exprel(710.0);
        assert!(value.is_finite() && value > 0.0, "exprel(710) = {value}");
        close(value / 3.146_472e305, 1.0, 1e-6);
        // The top of the band is `x − ln x = ln(MAX)`, at `716.4`: past there the quotient
        // overflows too and `+∞` is the right answer rather than a lost one.
        assert!(exprel(716.0).is_finite());
        assert_eq!(exprel(717.0), f64::INFINITY);
    }
}
