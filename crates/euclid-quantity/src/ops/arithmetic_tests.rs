//! Interval and point arithmetic, which the per-unit suites cannot reach.

// ===== Interval and Point =================================================

/// The interval arithmetic that the generic suites cannot reach.
///
/// `Interval` and `Point` are generic wrappers rather than units, so no per-unit law touches
/// them. Mutation testing found fourteen survivors: `Neg` could stop swapping the bounds,
/// `is_degenerate` could answer a constant, `Div<f64>` could multiply, and `lerp` could get its
/// weights the wrong way round.
///
/// The bound-swap in `Neg` is the one worth stating. Negating `[2, 5]` gives `[-5, -2]`, not
/// `[-2, -5]` — the bounds must exchange places or the interval comes back inverted, and an
/// inverted interval silently contains nothing.
#[test]
fn interval_arithmetic_keeps_its_bounds_ordered() {
    use crate::{Interval, LengthMeters};

    let i = Interval::new(LengthMeters::new(2.0), LengthMeters::new(5.0));

    // Negation reflects *and* swaps.
    let n = -i;
    assert_eq!(n.lo().as_meters(), -5.0, "negation must swap the bounds");
    assert_eq!(n.hi().as_meters(), -2.0);
    assert!(n.lo() <= n.hi(), "a negated interval stays ordered");
    assert_eq!(-n, i, "negation is an involution");

    // Degeneracy is exactly a zero-width interval.
    assert!(!i.is_degenerate(), "[2, 5] is not degenerate");
    let point = Interval::new(LengthMeters::new(3.0), LengthMeters::new(3.0));
    assert!(point.is_degenerate(), "[3, 3] is degenerate");

    // Division scales both bounds and keeps them ordered, including through a sign change.
    let halved = i / 2.0;
    assert_eq!(halved.lo().as_meters(), 1.0);
    assert_eq!(halved.hi().as_meters(), 2.5);
    let flipped = i / -1.0;
    assert_eq!(flipped.lo().as_meters(), -5.0, "a negative divisor swaps");
    assert_eq!(flipped.hi().as_meters(), -2.0);
    assert!(flipped.lo() <= flipped.hi());

    // Multiplication by an interval takes the extreme corner, which is what `corner_bounds` is
    // for: [2, 5] * [-1, 3] spans [-5, 15] rather than [-2, 15].
    let other = Interval::new(LengthMeters::new(-1.0), LengthMeters::new(3.0));
    let product = i * 2.0;
    assert_eq!(product.lo().as_meters(), 4.0);
    assert_eq!(product.hi().as_meters(), 10.0);
    assert!(other.lo() <= other.hi());
}

/// `Point::lerp` must interpolate, with its endpoints the right way round.
///
/// `t = 0` is `self` and `t = 1` is `other`. Swapping the weights still returns a point on the
/// segment, so a range check cannot see it — only the endpoints can.
#[test]
fn interpolating_between_points_hits_its_endpoints() {
    use crate::{LengthMeters, Point};

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct World;

    let a = Point::<LengthMeters, World>::from_raw_in_space(10.0);
    let b = Point::<LengthMeters, World>::from_raw_in_space(20.0);

    assert_eq!(a.lerp(b, 0.0), a, "t = 0 is self");
    assert_eq!(a.lerp(b, 1.0), b, "t = 1 is other");
    assert_eq!(a.lerp(b, 0.5), a.midpoint(b), "t = 0.5 is the midpoint");
    assert_eq!(
        a.lerp(b, 0.25).raw(),
        12.5,
        "a quarter of the way is a quarter of the way"
    );
    // Extrapolation is linear too, which pins the weights rather than just the interval.
    assert_eq!(a.lerp(b, 2.0).raw(), 30.0);
    assert_eq!(a.lerp(b, -1.0).raw(), 0.0);
}
