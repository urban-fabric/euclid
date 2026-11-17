//! Range properties.

use super::*;

// ===== Overflow and underflow rather than cancellation ====================

property! {
    #![config(settings("hypot").require("naive overflows", 50.0))]

    /// `hypot(x, y)` against `(x*x + y*y).sqrt()`.
    ///
    /// A different failure from cancellation: squaring a component above `2^512` overflows
    /// `f64` even when the norm is perfectly representable, so the naive form returns infinity
    /// for a finite answer.
    ///
    /// The exponent bound matters and is easy to get wrong — the first draft generated
    /// `magnitude(120, 300)`, reasoning in decimal. Those are *binary* exponents, so the
    /// largest value produced was about `4e90`, whose square is `1.6e181` and perfectly finite:
    /// the naive form never overflowed at all, and the coverage requirement caught it. Overflow
    /// needs `|x| > 2^512 ≈ 1.3e154`.
    ///
    /// Reference: the naive form *rescaled*. Factoring out the larger component is exactly what
    /// `hypot` is supposed to do, and the rescaled expression cannot overflow.
    fn hypot_avoids_intermediate_overflow(
        x in generate::float::magnitude(515, 1_020),
        y in generate::float::magnitude(515, 1_020),
    ) {
        let safe = Real::hypot(x, y);
        let naive = (x * x + y * y).sqrt();

        classify!("naive overflows", !naive.is_finite());
        classify!("naive survives", naive.is_finite());

        let m = x.abs().max(y.abs());
        let reference = m * ((x / m) * (x / m) + (y / m) * (y / m)).sqrt();

        check!(safe.is_finite(), "hypot overflowed on a representable norm");
        check_relative!(safe, reference, 1e-14, "hypot disagreed with the rescaled norm");
    }
}

property! {
    #![config(settings("hypot_underflow").require("naive underflows to zero", 50.0))]

    /// The underflow direction, which is the one people forget: squaring a component below
    /// `2^-537` flushes to zero, so the naive form returns zero for a nonzero norm.
    fn hypot_avoids_intermediate_underflow(
        x in generate::float::magnitude(-1_020, -540),
        y in generate::float::magnitude(-1_020, -540),
    ) {
        let safe = Real::hypot(x, y);
        let naive = (x * x + y * y).sqrt();

        classify!("naive underflows to zero", naive == 0.0);

        let m = x.abs().max(y.abs());
        let reference = m * ((x / m) * (x / m) + (y / m) * (y / m)).sqrt();

        check!(safe > 0.0, "hypot underflowed on a nonzero norm");
        check_relative!(safe, reference, 1e-14, "hypot disagreed with the rescaled norm");
    }
}

property! {
    #![config(settings("hypot3").require("naive overflows", 50.0))]

    /// `hypot3` has the same contract in three dimensions.
    fn hypot3_avoids_intermediate_overflow(
        x in generate::float::magnitude(515, 1_020),
        y in generate::float::magnitude(515, 1_020),
        z in generate::float::magnitude(515, 1_020),
    ) {
        let safe = Real::hypot3(x, y, z);
        let naive = (x * x + y * y + z * z).sqrt();
        classify!("naive overflows", !naive.is_finite());

        let m = x.abs().max(y.abs()).max(z.abs());
        let reference = m * ((x / m) * (x / m) + (y / m) * (y / m) + (z / m) * (z / m)).sqrt();

        check!(safe.is_finite(), "hypot3 overflowed on a representable norm");
        check_relative!(safe, reference, 1e-14, "hypot3 disagreed with the rescaled norm");
    }
}
