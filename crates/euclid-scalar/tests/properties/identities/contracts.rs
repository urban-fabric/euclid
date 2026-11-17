//! Contracts properties.

use super::*;

// ===== The tier contract, over generated input ============================

property! {
    #![config(settings("tier"))]

    /// `Exact` is the right comparison for the discrete parts of the surface, and this pins
    /// them: classification is a predicate, not an approximation.
    fn classification_is_exact(x in generate::float::any()) {
        relation::agree(
            "is_finite is the negation of is_nan-or-infinite",
            x,
            |v: f64| v.is_finite(),
            |v: f64| !(v.is_nan() || v.is_infinite()),
            &Exact,
        )?;
    }

    /// Negation is an involution — but only where `PartialEq` can see it.
    ///
    /// `Exact` compares with `==`, and `NaN != NaN`, so `involution` correctly reports that the
    /// law fails on `NaN`. That is not a bug in either the relation or in negation: it is the
    /// distinction between "the same value" and "the same bits", and `Exact` means the former.
    /// So the finite case goes through the relation and the `NaN` case is stated separately, in
    /// the comparison that can actually express it.
    fn negation_is_an_involution_on_the_finite_floats(x in generate::float::finite()) {
        relation::involution("negate", x, |v: f64| -v, &Exact)?;
    }

    /// And on `NaN`, negation round-trips *bitwise* — including the sign bit, which negation
    /// must flip rather than ignore.
    fn negating_nan_twice_restores_its_bits(_ in generate::just(0u8)) {
        let nan = f64::NAN;
        check!(
            (-(-nan)).to_bits() == nan.to_bits(),
            "double negation must restore NaN's bits"
        );
        check!(
            (-nan).is_sign_negative() != nan.is_sign_negative(),
            "negation must flip NaN's sign bit"
        );
    }
}
