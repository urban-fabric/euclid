//! The two route tables, as tables.
//!
//! Both predicates serve two orders apiece — four public functions dispatch on these two enums —
//! so a misroute moves several at once, and every arm returns a plausible oscillating value. That
//! is what a named route table is for, and what this asserts.

use crate::kernels::bessel::ordinary_first::{j0, j1};
use crate::kernels::bessel::ordinary_second::{y0, y1};

/// `J`'s six routes, over the whole line, and `Y`'s four over the positive half.
///
/// Both predicates serve two orders apiece — four public functions dispatch on these two enums —
/// so a misroute moves several at once, and every arm returns a plausible oscillating value.
/// `J₀` is even and `J₁` odd, which is why one magnitude ladder covers both.
#[test]
fn the_ordinary_bessel_routes_partition_their_domains() {
    use crate::kernels::bessel::ordinary_first::{HANKEL_MIN, J_SERIES_MAX, JRoute};
    use crate::kernels::bessel::ordinary_second::{Y_SERIES_MAX, YRoute};
    use JRoute::{Aliased, AtInfinity, AtOrigin, Hankel, Series, Undefined};

    for (x, want) in [
        (f64::NAN, Undefined),
        (0.0, AtOrigin),
        (-0.0, AtOrigin),
        (f64::INFINITY, AtInfinity),
        (f64::NEG_INFINITY, AtInfinity),
        // The ladder is on the magnitude, so both signs route alike.
        (1.0, Series),
        (-1.0, Series),
        (J_SERIES_MAX, Series),
        (-J_SERIES_MAX, Series),
        (J_SERIES_MAX.next_up(), Aliased),
        (10.0, Aliased),
        (HANKEL_MIN.next_down(), Aliased),
        (HANKEL_MIN, Hankel),
        (-HANKEL_MIN, Hankel),
        (1e300, Hankel),
    ] {
        assert_eq!(JRoute::for_argument(x), want, "J at {x:e}");
    }

    // `Y` is defined only for a positive argument, and both orders diverge to `−∞` at zero.
    for (x, want) in [
        (f64::NAN, YRoute::Value(f64::NAN)),
        (-1.0, YRoute::Value(f64::NAN)),
        (0.0, YRoute::Value(f64::NEG_INFINITY)),
        (f64::INFINITY, YRoute::Value(0.0)),
        (1.0, YRoute::Series),
        (Y_SERIES_MAX, YRoute::Series),
        (Y_SERIES_MAX.next_up(), YRoute::March),
        (10.0, YRoute::March),
        (HANKEL_MIN.next_down(), YRoute::March),
        (HANKEL_MIN, YRoute::Hankel),
        (1e300, YRoute::Hankel),
    ] {
        let got = YRoute::for_argument(x);
        match (got, want) {
            // A `NaN` payload is not equal to itself, so that arm compares by shape.
            (YRoute::Value(g), YRoute::Value(w)) if g.is_nan() && w.is_nan() => {}
            _ => assert_eq!(got, want, "Y at {x:e}"),
        }
    }

    // The parities the single ladder depends on, and the signed zero `J₁` carries through it.
    for x in [0.5_f64, 1.5, 3.0, 10.0, 25.0] {
        assert_eq!(j0(-x), j0(x), "J0 even at {x}");
        assert_eq!(j1(-x), -j1(x), "J1 odd at {x}");
    }
    assert_eq!(j0(0.0), 1.0);
    assert_eq!(j1(0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(j1(-0.0).to_bits(), (-0.0_f64).to_bits());
    assert!(y0(-1.0).is_nan() && y1(-1.0).is_nan());
    assert_eq!((y0(0.0), y1(0.0)), (f64::NEG_INFINITY, f64::NEG_INFINITY));
}
