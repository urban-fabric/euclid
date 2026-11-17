//! Transforms properties.

use super::*;

// ===== The exact primitives ===============================================

property! {
    #![config(settings("two_sum").cases(600).require("remainder is nonzero", 20.0))]

    /// `two_sum` must be **exact**: `s + e` is `a + b` with no error at all. Not "accurate to a
    /// few ulps" — exact, because every compensated algorithm in the crate is built on that
    /// guarantee and silently degrades if it only almost holds.
    ///
    /// Verifying exactness needs arithmetic wider than the thing under test, which is the whole
    /// difficulty. The first attempt here reconstructed the remainder as `(s - a) - b` and
    /// failed, correctly: those subtractions are themselves rounded in general, so the
    /// "reconstruction" was just another approximation. Sterbenz's lemma makes `s - a` exact
    /// only when the two are within a factor of two, which is not the interesting case.
    ///
    /// The fix is to generate operands that are **integers**, so `a + b` is exactly computable
    /// in `i128` and the comparison is against true arithmetic rather than against more `f64`.
    /// Integer operands still exercise the hard part — the magnitudes differ by up to 2^53, so
    /// the sum rounds and the remainder is genuinely nonzero.
    fn two_sum_is_exact_on_integer_operands(
        a_bits in generate::int(-(1i64 << 53), 1i64 << 53),
        b_shift in generate::int(0, 60),
    ) {
        // `a` is an integer; `b` is an integer scaled far enough away that `a + b` must round.
        #[allow(clippy::cast_precision_loss)]
        let a = a_bits as f64;
        let b = (a_bits.rotate_left(17) % (1i64 << 40)) as f64 * 2.0_f64.powi(b_shift as i32);
        if !a.is_finite() || !b.is_finite() {
            return Ok(());
        }

        let (sum, error) = Real::two_sum(a, b);
        if !sum.is_finite() {
            check_ulps!(error, 0.0, 0, "a non-finite sum must carry no remainder");
            return Ok(());
        }

        classify!("remainder is nonzero", error != 0.0);
        classify!("remainder is zero", error == 0.0);

        // Both `sum` and `error` are integers here (sums of integers, exactly representable or
        // not), so the true total can be compared in i128 with no rounding anywhere.
        let exact_total = i128::from(a_bits) + (b as i128);
        let reconstructed = (sum as i128) + (error as i128);
        check!(
            reconstructed == exact_total,
            "two_sum({a}, {b}) = ({sum}, {error}) reconstructs {reconstructed}, not {exact_total}"
        );
    }

}

property! {
    // Its own block: the requirement above names `two_sum`'s label, and a block-level config
    // applies to every property in it. Sharing one block made this property fail for having
    // never applied a label it does not use, which is the footgun worth keeping separated.
    #![config(settings("two_product"))]

    /// `two_product` must likewise be exact: `p + e` is `a * b`, to the last bit.
    ///
    /// Here there *is* a wider primitive available — `fma(a, b, -p)` is the exact remainder by
    /// definition — so it serves as the reference directly.
    fn two_product_is_exact(
        a in generate::float::magnitude(-200, 200),
        b in generate::float::magnitude(-200, 200),
    ) {
        let (product, error) = Real::two_product(a, b);
        if !product.is_finite() || (error != 0.0 && !error.is_finite()) {
            return Ok(());
        }
        let exact_remainder = Real::mul_add(a, b, -product);
        check_ulps!(error, exact_remainder, 0, "two_product lost part of the remainder");
    }
}
