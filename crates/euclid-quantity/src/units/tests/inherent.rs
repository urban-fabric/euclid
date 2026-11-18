//! Accessors, predicates and display methods compared with the underlying `f64`.
//! The family cases exercise the methods each unit exposes. Constructors can reject NaN
//! in debug builds, so the probes use finite values, signed zeros and infinities.

/// Values every unit must handle, including both zeros and both infinities.
const PROBES: [f64; 9] = [
    0.0,
    -0.0,
    1.0,
    -2.5,
    7.5,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::MAX,
    f64::MIN_POSITIVE,
];

/// The accessor half, which every f64 unit has: `as_x` inverts `new` and agrees with
/// `Measure::raw`.
macro_rules! accessor_agrees {
    ($ty:ty, $as:ident) => {{
        for v in crate::units::tests::inherent::PROBES {
            let q = <$ty>::new(v);
            let label = concat!(stringify!($ty), "::", stringify!($as));
            assert_eq!(q.$as(), v, "{label}({v:e}) must invert new");
            assert_eq!(
                q.$as(),
                crate::Measure::raw(&q),
                "{label}({v:e}) must agree with Measure::raw"
            );
        }
        assert_eq!(<$ty>::ZERO.$as(), 0.0, "ZERO must be zero");
    }};
}

/// The sign predicates, which must mean what their names say.
macro_rules! sign_predicates_agree {
    ($ty:ty, $as:ident) => {{
        for v in crate::units::tests::inherent::PROBES {
            let q = <$ty>::new(v);
            let label = stringify!($ty);
            assert_eq!(
                q.is_positive(),
                v > 0.0,
                "{label}: is_positive({v:e}) must mean strictly greater than zero"
            );
            assert_eq!(
                q.is_non_negative(),
                v >= 0.0,
                "{label}: is_non_negative({v:e}) must include zero"
            );
            assert_eq!(
                q.is_zero(),
                v == 0.0,
                "{label}: is_zero({v:e}) must mean exactly zero"
            );
        }
        assert!(<$ty>::ZERO.is_zero());
        assert!(!<$ty>::ZERO.is_positive(), "zero is not positive");
        assert!(<$ty>::ZERO.is_non_negative(), "zero is non-negative");
    }};
}

/// `is_finite`, which must reject both infinities.
macro_rules! finiteness_agrees {
    ($ty:ty) => {{
        for v in crate::units::tests::inherent::PROBES {
            let q = <$ty>::new(v);
            assert_eq!(
                q.is_finite(),
                v.is_finite(),
                concat!(stringify!($ty), ": is_finite must agree with the raw value"),
            );
        }
        assert!(
            !<$ty>::new(f64::INFINITY).is_finite(),
            "an infinity is not finite"
        );
    }};
}

/// `abs`, where the unit offers it.
macro_rules! abs_agrees {
    ($ty:ty, $as:ident) => {{
        for v in crate::units::tests::inherent::PROBES {
            assert_eq!(
                <$ty>::new(v).abs().$as(),
                v.abs(),
                concat!(stringify!($ty), "::abs must agree with f64::abs"),
            );
        }
    }};
}

/// `Display` must distinguish values and name its unit.
///
/// Every unit's `Display` is a one-line delegation to the shared
/// `crate::conversion::fmt_quantity`, and mutation testing found all sixty-four replaceable with
/// a body that writes nothing and returns `Ok(())`. The shared helper is well covered; its
/// sixty-four delegating callers were not — the same split as the operator bridge.
///
/// Asserted format-agnostically, because the point is the delegation rather than the layout: two
/// different values must render differently, and a unit with a symbol must show it. A body that
/// writes nothing fails the first; one that writes only the symbol fails it too.
macro_rules! display_distinguishes {
    ($ty:ty) => {{
        let one = <$ty>::new(1.0).to_string();
        let two = <$ty>::new(2.0).to_string();
        assert!(
            !one.is_empty(),
            concat!(stringify!($ty), " rendered to nothing")
        );
        assert_ne!(
            one, two,
            concat!(stringify!($ty), ": Display must distinguish values")
        );
        if !<$ty as euclid_units::Measure>::UNIT_NAME.is_empty() {
            assert!(
                one.contains(<$ty as euclid_units::Measure>::UNIT_NAME),
                concat!(stringify!($ty), ": Display must name its unit"),
            );
        }
    }};
}

/// `is_greater_than_epsilon` is a strict comparison against its argument, not against zero.
macro_rules! epsilon_predicate_agrees {
    ($ty:ty) => {{
        for v in crate::units::tests::inherent::PROBES {
            for eps in [0.0_f64, 1e-9, 1.0, -1.0] {
                assert_eq!(
                    <$ty>::new(v).is_greater_than_epsilon(eps),
                    v > eps,
                    concat!(
                        stringify!($ty),
                        ": is_greater_than_epsilon must be a strict comparison against epsilon"
                    ),
                );
            }
        }
    }};
}

/// A unit carrying the whole inherent surface.
macro_rules! full_surface {
    ($name:ident, $ty:ty, $as:ident) => {
        #[test]
        fn $name() {
            accessor_agrees!($ty, $as);
            sign_predicates_agree!($ty, $as);
            finiteness_agrees!($ty);
            abs_agrees!($ty, $as);
            display_distinguishes!($ty);
        }
    };
}

mod chemistry;
mod dimensionless;
mod electromagnetism;
mod geometry;
mod kinematics;
mod mechanics;
mod thermal;

/// Only three units carry `is_greater_than_epsilon`, so it gets one test rather than a
/// macro arm per unit.
#[test]
fn the_epsilon_predicate_is_a_strict_comparison() {
    epsilon_predicate_agrees!(crate::Factor);
    epsilon_predicate_agrees!(crate::SpeedKph);
    epsilon_predicate_agrees!(crate::TimeSeconds);
}

/// The remaining per-unit helpers on the f64 units: the `f32` narrowing accessors, the positivity
/// assertion, and the two predicates the generated suites do not know about.
#[test]
fn the_per_unit_helpers_agree_with_the_raw_value() {
    use crate::{LengthMeters, SpeedKph, TimeSeconds};

    for v in [0.0_f64, 1.5, -2.5, 1e20] {
        #[allow(clippy::cast_possible_truncation)]
        {
            assert_eq!(
                LengthMeters::new(v).to_f32_meters(),
                v as f32,
                "to_f32_meters must narrow the raw value"
            );
            assert_eq!(SpeedKph::new(v).to_f32_kph(), v as f32);
        }
    }

    // `is_greater_than_epsilon` is a strict comparison against its argument.
    for v in [0.0_f64, 1e-12, 1.0, -1.0] {
        for eps in [0.0_f64, 1e-9, 1.0, -1.0] {
            assert_eq!(
                LengthMeters::new(v).is_greater_than_epsilon(eps),
                v > eps,
                "is_greater_than_epsilon({v}, {eps}) must be strict"
            );
        }
    }

    // `is_infinite` must reject finite values and accept both infinities.
    assert!(TimeSeconds::new(f64::INFINITY).is_infinite());
    assert!(TimeSeconds::new(f64::NEG_INFINITY).is_infinite());
    assert!(!TimeSeconds::new(0.0).is_infinite());
    assert!(!TimeSeconds::new(1e300).is_infinite());

    // `assert_positive` returns its input unchanged for a positive value — it is a debug
    // assertion, so the contract in release is the identity.
    assert_eq!(
        LengthMeters::new(2.5).assert_positive(),
        LengthMeters::new(2.5),
        "assert_positive must return its input"
    );
    assert_eq!(
        TimeSeconds::new(1.0).assert_positive(),
        TimeSeconds::new(1.0)
    );
    assert_eq!(SpeedKph::new(30.0).assert_positive(), SpeedKph::new(30.0));
}
