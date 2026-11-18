//! Angle laws anchored at quarter, half and full turns.

use crate::AngleRadians;

/// The angle surface: turns, the degree/radian relationship, the trigonometry, and
/// normalisation.
///
/// Seventeen survivors lived here, more than any other single file at this point. The cause is
/// that `AngleRadians` reaches the generic suites through `quantity_laws` and the dimensional
/// tests, and neither says anything about the `Angle` tier's own methods — `sin`, `cos`, `tan`,
/// `half_turn`, `normalized_signed`. `sin` could return `0.0` and `cos` `1.0` for every angle.
///
/// The anchors are the constants, as with the temperature scales: a half turn *is* pi, a full
/// turn *is* tau, and 180 degrees *is* a half turn. Those come from mathematics rather than from
/// the implementation, which is what makes them worth asserting.
#[test]
fn the_angle_tier_matches_the_constants_and_the_trigonometry() {
    use core::f64::consts::{PI, TAU};
    use euclid_units::Angle;

    const TOL: f64 = 1e-12;
    let close = |got: f64, want: f64, law: &str| {
        assert!((got - want).abs() <= TOL, "{law}: got {got}, want {want}");
    };

    // The turns are the constants they are named for.
    close(AngleRadians::half_turn().radians(), PI, "a half turn is pi");
    close(
        AngleRadians::full_turn().radians(),
        TAU,
        "a full turn is tau",
    );

    // Degrees and radians are the same angle, anchored at the quarter turns.
    for (degrees, radians) in [
        (0.0, 0.0),
        (90.0, PI / 2.0),
        (180.0, PI),
        (270.0, 3.0 * PI / 2.0),
        (360.0, TAU),
        (-90.0, -PI / 2.0),
    ] {
        close(
            AngleRadians::from_degrees(degrees).radians(),
            radians,
            "from_degrees agrees with radians",
        );
        close(
            AngleRadians::from_radians(radians).degrees(),
            degrees,
            "from_radians agrees with degrees",
        );
    }

    // The trigonometry is the trigonometry.
    for radians in [0.0_f64, 0.5, 1.0, -1.0, PI / 4.0, PI, 2.5] {
        let a = AngleRadians::from_radians(radians);
        close(a.radians(), radians, "from_radians then radians");
        close(a.sin(), radians.sin(), "sin agrees with f64::sin");
        close(a.cos(), radians.cos(), "cos agrees with f64::cos");
        close(a.tan(), radians.tan(), "tan agrees with f64::tan");
        #[allow(clippy::cast_possible_truncation)]
        {
            assert_eq!(
                a.to_f32_radians(),
                radians as f32,
                "to_f32_radians must narrow the radian value"
            );
        }
    }

    // Normalisation lands in the documented range and preserves the angle itself — which is the
    // law that catches a wrap in the wrong direction, since the range check alone does not.
    for radians in [0.0_f64, 0.5, PI, -PI, 3.0 * PI, -3.0 * PI, 100.0, -100.0] {
        let signed = AngleRadians::from_radians(radians).normalized_signed();
        assert!(
            signed.radians() > -PI - TOL && signed.radians() <= PI + TOL,
            "normalized_signed({radians}) = {} is outside (-pi, pi]",
            signed.radians()
        );
        close(
            signed.sin(),
            radians.sin(),
            "normalized_signed must preserve the angle",
        );

        let unsigned = AngleRadians::from_radians(radians).normalized_unsigned();
        assert!(
            unsigned.radians() >= -TOL && unsigned.radians() < TAU + TOL,
            "normalized_unsigned({radians}) = {} is outside [0, tau)",
            unsigned.radians()
        );
        close(
            unsigned.cos(),
            radians.cos(),
            "normalized_unsigned must preserve the angle",
        );
    }

    // The **half-open end** of the range: pi normalises to pi, not to -pi. The range check
    // above cannot see this, because -pi is inside `(-pi, pi]` by a hair under the tolerance —
    // only an exact assertion can, and mutation testing found the comparison unpinned without
    // it. `(-pi, pi]` is closed at the top, so a half turn stays a positive half turn.
    close(
        AngleRadians::from_radians(PI).normalized_signed().radians(),
        PI,
        "pi normalises to pi, not -pi",
    );
    close(
        AngleRadians::from_radians(-PI)
            .normalized_signed()
            .radians(),
        PI,
        "-pi also normalises to pi, since the range is closed at the top",
    );

    // And the two normalisations agree with each other modulo a turn.
    for radians in [0.5_f64, 4.0, -4.0, 7.0] {
        let s = AngleRadians::from_radians(radians)
            .normalized_signed()
            .radians();
        let u = AngleRadians::from_radians(radians)
            .normalized_unsigned()
            .radians();
        assert!(
            (s - u).abs() <= TOL || (s - u).abs() - TAU <= TOL,
            "the two normalisations differ by something other than a turn: {s} vs {u}"
        );
    }
}
