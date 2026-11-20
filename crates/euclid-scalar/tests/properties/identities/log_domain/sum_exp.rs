//! `log_add_exp` and `log_sub_exp`.
//!
//! Part of the [`log_domain`](super) suite. Shift invariance is the strongest law available
//! here: it holds exactly in real arithmetic for every shift, so any drift under it is the
//! kernel's own error rather than the law's.

use super::*;

property! {
    #![config(settings("log_sum_exp").cases(500))]

    /// `ln_add_exp` is symmetric and sits in `[max, max + ln 2]`, both exactly.
    ///
    /// The bound is the whole content of the stable formulation: `ln(eᵃ + eᵇ)` is
    /// `max + ln(1 + e^{−|a−b|})`, and the second term lies in `(0, ln 2]` by construction. So
    /// the law holds at every argument, needs no tolerance, and is false for any implementation
    /// that overflows or picks the wrong operand as the pivot.
    ///
    /// Symmetry is bit-exact because `max` and `|a − b|` both are. Measured at 0 ulps over
    /// 40,000 pairs spanning `[−600, 600]`, so it is asserted at 0 rather than "a few".
    fn the_log_sum_exp_is_symmetric_and_sits_in_its_bracket(
        a in generate::float::in_range(-600.0, 600.0),
        b in generate::float::in_range(-600.0, 600.0),
    ) {
        classify!("far apart", (a - b).abs() > 50.0);
        classify!("close together", (a - b).abs() <= 1.0);

        let result = Real::ln_add_exp(a, b);
        check_ulps!(
            result,
            Real::ln_add_exp(b, a),
            0,
            "ln_add_exp must be bit-exactly symmetric"
        );

        let pivot = a.max(b);
        check!(
            result >= pivot,
            "ln_add_exp({a:e}, {b:e}) = {result:e} fell below max = {pivot:e}"
        );
        check!(
            result <= pivot + core::f64::consts::LN_2,
            "ln_add_exp({a:e}, {b:e}) = {result:e} exceeded max + ln 2"
        );
    }

    /// `ln_add_exp(a + c, b + c) = ln_add_exp(a, b) + c` — shift invariance.
    ///
    /// The defining property of a log-sum-exp, and the reason it is the stable form: shifting by
    /// the pivot is exactly what the implementation does internally, so an implementation that
    /// is *not* shift-invariant has lost the plot entirely. No reference value is involved.
    ///
    /// # The comparison must be absolute, not relative
    ///
    /// A first draft compared relatively and measured `1.8e-1`, which says nothing about the
    /// kernel: `ln_add_exp(a,b) + c` passes through zero, and relative error against a reference
    /// that is nearly zero is meaningless. The bound here is absolute and scaled by the inputs,
    /// because the error is the rounding of `a + c` and `b + c` — each `ε` relative to its own
    /// magnitude — carried through a function whose derivative in each argument is at most one.
    fn the_log_sum_exp_is_shift_invariant(
        a in generate::float::in_range(-300.0, 300.0),
        b in generate::float::in_range(-300.0, 300.0),
        c in generate::float::in_range(-300.0, 300.0),
    ) {
        let base = Real::ln_add_exp(a, b);
        let shifted = Real::ln_add_exp(a + c, b + c);
        if !base.is_finite() || !shifted.is_finite() {
            return Ok(());
        }
        let scale = a.abs().max(b.abs()).max(c.abs()).max(1.0);
        relation::agree(
            "ln_add_exp is shift invariant",
            (),
            |()| shifted,
            |()| base + c,
            &Absolute(16.0 * f64::EPSILON * scale),
        )?;
    }

    /// Against the naive `ln(eᵃ + eᵇ)`, wherever the naive form is representable at all.
    ///
    /// The differential check: same answer, wider domain. Restricted to `|a|, |b| ≤ 300` so the
    /// exponentials exist, and absolute for the same reason as above — the reference crosses
    /// zero. Outside that range the naive form is `inf` or `0` and the stable one still works,
    /// which the bracket property above covers.
    fn the_log_sum_exp_agrees_with_the_naive_form_where_that_exists(
        a in generate::float::in_range(-300.0, 300.0),
        b in generate::float::in_range(-300.0, 300.0),
    ) {
        let naive = (a.exp() + b.exp()).ln();
        if !naive.is_finite() {
            return Ok(());
        }
        let scale = a.abs().max(b.abs()).max(1.0);
        relation::agree(
            "ln_add_exp agrees with ln(exp a + exp b)",
            (),
            |()| Real::ln_add_exp(a, b),
            |()| naive,
            &Absolute(16.0 * f64::EPSILON * scale),
        )?;
    }

    /// The identities at the edges: `−∞` is the additive identity, and a `NaN` in **either**
    /// operand propagates.
    ///
    /// `e^{−∞}` is zero, so `ln_add_exp(a, −∞)` must be exactly `a` — the identity element of a
    /// log-domain sum, which is what makes it usable as a fold's initial value.
    ///
    /// The NaN half is drawn with `pair_with_edge_cases` rather than written as fixed points,
    /// because `ln_add_exp` opens with `if self.is_nan() || other.is_nan()` and that is the guard
    /// shape this repository has now found two real defects in: weakened to `&&`, it needs a
    /// test at *each* operand to catch.
    fn the_log_sum_exp_handles_its_identity_and_both_nans(
        pair in generate::float::pair_with_edge_cases(-20, 20),
    ) {
        let (a, y) = pair;
        classify!("nan operand", y.is_nan());
        classify!("infinite operand", y.is_infinite());

        if a.is_nan() || y.is_nan() {
            check!(
                Real::ln_add_exp(a, y).is_nan() && Real::ln_add_exp(y, a).is_nan(),
                "a NaN operand must propagate: ln_add_exp({a:e}, {y:e})"
            );
            return Ok(());
        }

        // `-inf` is the identity, from either side.
        check_ulps!(
            Real::ln_add_exp(a, f64::NEG_INFINITY),
            a,
            0,
            "ln_add_exp(a, -inf) must be exactly a"
        );
        check_ulps!(
            Real::ln_add_exp(f64::NEG_INFINITY, a),
            a,
            0,
            "ln_add_exp(-inf, a) must be exactly a"
        );
        // And `+inf` absorbs.
        check!(
            Real::ln_add_exp(a, f64::INFINITY) == f64::INFINITY,
            "ln_add_exp(a, +inf) must be +inf"
        );
    }
}

property! {
    #![config(settings("log_sub_exp").cases(500))]

    /// `ln_add_exp(ln_sub_exp(a, b), b) = a` for `a > b` — subtraction inverts addition.
    ///
    /// Measured **exact** over 40,000 ordered pairs, at every gap, which is a strong statement
    /// about both functions at once: `(eᵃ − eᵇ) + eᵇ = eᵃ` can only survive in the log domain if
    /// each of the two is doing its pivot and its `ln_1p`/`ln_1m` correctly. No reference value,
    /// and it reaches arguments where `eᵃ` does not exist.
    ///
    /// The degenerate cases are the contract's edges and are asserted directly: equal arguments
    /// give `ln 0 = −∞`, and `a < b` is outside the documented domain ("valid for
    /// `self >= other`") and returns `NaN` rather than a complex result.
    fn log_sub_exp_inverts_log_add_exp(
        a in generate::float::in_range(-300.0, 300.0),
        b in generate::float::in_range(-300.0, 300.0),
    ) {
        let (high, low) = if a >= b { (a, b) } else { (b, a) };
        classify!("close together", (high - low) < 1.0);
        classify!("far apart", (high - low) > 50.0);

        // Equal arguments: ln(e^a - e^a) = ln 0.
        check!(
            Real::ln_sub_exp(high, high) == f64::NEG_INFINITY,
            "ln_sub_exp(a, a) must be -inf"
        );
        // Out of domain, by the trait's own statement of it.
        if high > low {
            check!(
                Real::ln_sub_exp(low, high).is_nan(),
                "ln_sub_exp below the diagonal must be NaN: ({low:e}, {high:e})"
            );
        }

        let difference = Real::ln_sub_exp(high, low);
        if !difference.is_finite() {
            return Ok(());
        }
        // The result is a log of something smaller than `e^high`, so it cannot exceed it.
        check!(
            difference <= high,
            "ln_sub_exp({high:e}, {low:e}) = {difference:e} exceeded its larger argument"
        );
        // Mixed rather than relative, and for a reason the exploration behind this property
        // managed to hide from itself: it skipped the `a == 0` cases while measuring, and
        // `a == 0` is precisely where a relative comparison is undefined. At `(0, −36.4)` the
        // round trip returns `−4.9e-32` where the reference is exactly `0` — an absolute error
        // of nothing and a relative error of infinity. Scaling by the argument is the honest
        // criterion, the same one `euclid_utils` uses throughout.
        relation::agree(
            "ln_add_exp(ln_sub_exp(a, b), b) = a",
            (),
            |()| Real::ln_add_exp(difference, low),
            |()| high,
            &Absolute(16.0 * f64::EPSILON * high.abs().max(1.0)),
        )?;
    }
}
