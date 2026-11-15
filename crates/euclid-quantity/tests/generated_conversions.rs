//! Unit conversion as a metamorphic relation: convert away and back, and arrive
//! where you started.
//!
//! `UnitOfMeasure::convert` is one generic function keyed on a dimension, so
//! every unit newtype in the crate reaches the same two multiplications —
//! `in_primary` then `value_from_primary`. That makes the round trip the right
//! thing to test, and makes it worth testing over generated magnitudes rather
//! than the round numbers a hand-written case reaches for.
//!
//! A conversion needs no oracle: nobody has to know how many feet are in a
//! kilometre for `km → ft → km` to be checkable. That is the whole appeal, and
//! it is why `1 km == 3280.84 ft` (the doc example) and this suite are testing
//! different things. The constant proves the factor is right; the round trip
//! proves the *protocol* is, at every magnitude, including the ones where the
//! two multiplications stop cancelling.
//!
//! ## What a round trip can and cannot promise
//!
//! `value_from_primary(in_primary(x))` is `x · k · (1/k)`, and in `f64` that is
//! not `x`. The reciprocal is rounded, so the product is within a couple of ULP
//! and not exact — which is why every comparison here is `check_ulps!` rather
//! than `==`. Two ULP is the honest bound for two multiplications by a rounded
//! constant, and asserting equality would be asserting something false about
//! floating point rather than something true about units.
//!
//! The exception is a conversion whose factor is a power of two or one, where
//! the round trip *is* exact. Those are not special-cased: a tolerance of two
//! ULP admits them, and carving out an exact case would be asserting the factor
//! rather than the law.
//!
//! ## Affine units are a different law, and that is the point
//!
//! Temperature is the one dimension here that is **affine** rather than linear:
//! `°C → K` adds an offset, so `0 °C` is not `0 K` and scaling a temperature is
//! meaningless. A conversion protocol that treated it as a bare factor would
//! round-trip correctly for `0` and fail everywhere else, which is exactly the
//! bug a generated magnitude finds and a fixture at freezing point does not.
//!
//! It gets the same round-trip *law* as the linear units but not the same
//! *bound*, and the difference is a real statement about conditioning rather
//! than a concession. An affine round trip travels through the primary unit, so
//! `1 °F` becomes `-17.2 °C` and comes back through `-31 + 32`. The absolute
//! error of the large intermediate lands on a small result: the round trip
//! returns `0.9999999999999772`, which is **205 ULP** of `1.0` and well under
//! one ULP of the `255.9 K` it passed through.
//!
//! That is correct arithmetic. The condition number of the trip is
//! `|F - 32| / |F|`, which is 31 at `F = 1` and unbounded as `F` approaches the
//! offset — so no fixed ULP count on the *result* can be honest. The bound below
//! is taken against the largest magnitude the value reaches anywhere along the
//! trip, which is the quantity the absolute error is proportional to.
//!
//! Getting that denominator right took two attempts, and the second one is the
//! interesting half. Bounding against the *primary unit* still failed, on
//! `9.3e-10 K → °C → K`: Kelvin is itself the primary unit, so the primary-unit
//! magnitude is the tiny start value, while the trip actually passes through
//! `-273.15 °C`. The large value is the one *in between*, which neither endpoint
//! nor the primary unit names. A 2-ULP bound on the result looked tighter and
//! was simply false; this one is tight against the right denominator.

use euclid_property::{Config, Gen, check, check_ulps, classify, generate, property};
use euclid_quantity::{
    AngleDegrees, AngleRadians, LengthFeet, LengthKm, LengthMeters, MassGrams, MassKilograms,
    MassMilligrams, TemperatureCelsius, TemperatureFahrenheit, TemperatureKelvin, TimeHours,
    TimeMillis, TimeMinutes, TimeSeconds, UnitOfMeasure,
};
use euclid_units::Measure;

fn settings(suite: &str) -> Config {
    Config::default()
        .cases(400)
        .regressions(format!("tests/regressions/{suite}.txt"))
}

/// Magnitudes spanning the range a quantity realistically takes, by decade.
///
/// Log-uniform rather than linear because the interesting behaviour of a
/// multiply-by-constant is spread across the exponent, not across an interval.
/// The exponents are **binary**, so this is roughly `1e-9` to `1e9`.
fn magnitude() -> Gen<f64> {
    generate::float::magnitude(-30, 30)
}

/// Magnitudes plus the values that are their own special case.
///
/// Zero and one are where an affine conversion and a linear one disagree, and
/// where a factor applied in the wrong direction still looks right.
fn value() -> Gen<f64> {
    generate::weighted(vec![
        (2, generate::just(0.0)),
        (2, generate::just(1.0)),
        (2, generate::just(-1.0)),
        (12, magnitude()),
    ])
}

/// Label a magnitude by decade band, so the coverage floors below can demand
/// that the extremes were actually reached.
fn band(v: f64) -> &'static str {
    let m = v.abs();
    if m == 0.0 {
        "zero"
    } else if m < 1e-6 {
        "tiny"
    } else if m < 1e6 {
        "ordinary"
    } else {
        "huge"
    }
}

/// Few enough ULP to catch a wrong factor, against the magnitude the error is
/// proportional to. See the module header on affine conditioning.
const AFFINE_ULPS: f64 = 4.0;

/// The affine round trip: same law, bound against the primary-unit magnitude.
macro_rules! affine_round_trip {
    ($from:ty => $via:ty, $raw:expr, $label:expr) => {{
        let original = <$from>::new($raw);
        let there: $via = original.convert();
        let back: $from = there.convert();
        // The error is a few ULP of the largest magnitude the value takes
        // *anywhere along the trip*, which is neither endpoint. Going from
        // `9.3e-10 K` to Celsius passes through `-273.15`, and it is that
        // intermediate the absolute error is proportional to — so the
        // intermediate's own magnitude has to be in the denominator. Bounding
        // against the primary unit alone is not enough: for Kelvin the primary
        // unit *is* the start, and the large value is the one in between.
        let primary = original
            .raw()
            .abs()
            .max(there.raw().abs())
            .max(original.in_primary().abs())
            .max(1.0);
        let slack = AFFINE_ULPS * f64::EPSILON * primary;
        check!(
            (back.raw() - original.raw()).abs() <= slack,
            "{}: {} -> {} -> {} returned {} (slack {:e}, primary magnitude {})",
            $label,
            $raw,
            stringify!($via),
            stringify!($from),
            back.raw(),
            slack,
            primary
        );
    }};
}

/// The round trip, as a macro so each pair reads as one line.
///
/// A generic function would need `UnitOfMeasure<Dim = D>` bounds spelled out at
/// every call; the macro says the same thing in the place the reader is looking.
macro_rules! round_trip {
    ($from:ty => $via:ty, $raw:expr, $label:expr) => {{
        let original = <$from>::new($raw);
        let there: $via = original.convert();
        let back: $from = there.convert();
        check_ulps!(
            back.raw(),
            original.raw(),
            2,
            "{}: {} -> {} -> {} lost the value",
            $label,
            $raw,
            stringify!($via),
            stringify!($from)
        );
    }};
}

// ===== Length =============================================================

property! {
    #![config(
        settings("length")
            .require("tiny", 1.0)
            .require("huge", 1.0)
            .require("ordinary", 1.0)
    )]

    /// Metres through every other length unit and back.
    fn length_round_trips_through_every_unit(v in value()) {
        classify!(band(v));
        round_trip!(LengthMeters => LengthKm, v, "m/km");
        round_trip!(LengthMeters => LengthFeet, v, "m/ft");
        round_trip!(LengthKm => LengthFeet, v, "km/ft");
        round_trip!(LengthFeet => LengthMeters, v, "ft/m");
    }

    /// Converting through an intermediate unit agrees with converting direct.
    ///
    /// The associativity of the protocol, and a sharper claim than the round
    /// trip: a round trip passes if a factor is wrong in both directions,
    /// because the error cancels. Going `km → m → ft` versus `km → ft` cannot
    /// cancel, so it catches a factor that is simply the wrong number.
    fn converting_via_an_intermediate_unit_agrees_with_direct(v in value()) {
        classify!(band(v));
        let km = LengthKm::new(v);

        let direct: LengthFeet = km.convert();
        let meters: LengthMeters = km.convert();
        let via_meters: LengthFeet = meters.convert();

        // Three multiplications against two, so one more ULP of room.
        check_ulps!(
            via_meters.raw(),
            direct.raw(),
            3,
            "km -> ft direct gave {} but km -> m -> ft gave {}",
            direct.raw(),
            via_meters.raw()
        );
    }
}

// ===== Time ===============================================================

property! {
    #![config(settings("time").require("tiny", 1.0).require("huge", 1.0))]

    /// Seconds through the larger and smaller time units and back.
    fn time_round_trips_through_every_unit(v in value()) {
        classify!(band(v));
        round_trip!(TimeSeconds => TimeMillis, v, "s/ms");
        round_trip!(TimeSeconds => TimeMinutes, v, "s/min");
        round_trip!(TimeSeconds => TimeHours, v, "s/h");
        round_trip!(TimeHours => TimeMillis, v, "h/ms");
    }

    /// The sexagesimal chain composes: `h → min → s` agrees with `h → s`.
    ///
    /// Worth its own property because 60 and 3600 are the factors most likely
    /// to be typed wrong, and a round trip would hide it.
    fn the_sexagesimal_chain_composes(v in value()) {
        classify!(band(v));
        let hours = TimeHours::new(v);
        let direct: TimeSeconds = hours.convert();
        let minutes: TimeMinutes = hours.convert();
        let via_minutes: TimeSeconds = minutes.convert();
        check_ulps!(
            via_minutes.raw(),
            direct.raw(),
            3,
            "h -> s direct gave {} but h -> min -> s gave {}",
            direct.raw(),
            via_minutes.raw()
        );
    }
}

// ===== Mass ===============================================================

property! {
    #![config(settings("mass").require("tiny", 1.0).require("huge", 1.0))]

    /// The metric mass ladder round-trips.
    ///
    /// Every factor here is a power of ten, so these are the conversions where
    /// the round trip comes closest to exact — and the ones where a tolerance
    /// that was too loose would hide a real error.
    fn mass_round_trips_through_every_unit(v in value()) {
        classify!(band(v));
        round_trip!(MassKilograms => MassGrams, v, "kg/g");
        round_trip!(MassKilograms => MassMilligrams, v, "kg/mg");
        round_trip!(MassGrams => MassMilligrams, v, "g/mg");
    }
}

// ===== Angle ==============================================================

property! {
    #![config(settings("angle").require("tiny", 1.0).require("huge", 1.0))]

    /// Degrees and radians round-trip.
    ///
    /// The factor is `π/180`, which is irrational and therefore rounded — so
    /// this is the pair where a round trip genuinely cannot be exact, and the
    /// two-ULP bound is doing real work rather than being generous.
    fn angle_round_trips_between_degrees_and_radians(v in value()) {
        classify!(band(v));
        round_trip!(AngleDegrees => AngleRadians, v, "deg/rad");
        round_trip!(AngleRadians => AngleDegrees, v, "rad/deg");
    }
}

// ===== Temperature: the affine one ========================================

property! {
    #![config(
        settings("temperature")
            .require("tiny", 1.0)
            .require("ordinary", 1.0)
            .require("huge", 1.0)
    )]

    /// Temperature round-trips through every scale.
    ///
    /// The affine dimension: `°C → K` adds an offset rather than scaling, so a
    /// protocol that treated it as a bare factor would pass at `0` and fail
    /// everywhere else. A generated magnitude is what separates the two; a
    /// fixture at freezing point is exactly the case that cannot.
    fn temperature_round_trips_through_every_scale(v in value()) {
        classify!(band(v));
        affine_round_trip!(TemperatureKelvin => TemperatureCelsius, v, "K/C");
        affine_round_trip!(TemperatureKelvin => TemperatureFahrenheit, v, "K/F");
        affine_round_trip!(TemperatureCelsius => TemperatureFahrenheit, v, "C/F");
        affine_round_trip!(TemperatureFahrenheit => TemperatureCelsius, v, "F/C");
    }

    /// The Celsius/Fahrenheit chain composes through Kelvin.
    ///
    /// For an affine conversion this is the claim with teeth: offsets have to
    /// compose as well as factors, and adding an offset twice (or dropping one
    /// on the way through the primary unit) survives a round trip and dies here.
    fn the_affine_chain_composes_through_the_primary_unit(v in value()) {
        classify!(band(v));
        let celsius = TemperatureCelsius::new(v);
        let direct: TemperatureFahrenheit = celsius.convert();
        let kelvin: TemperatureKelvin = celsius.convert();
        let via_kelvin: TemperatureFahrenheit = kelvin.convert();
        let primary = v
            .abs()
            .max(kelvin.raw().abs())
            .max(direct.raw().abs())
            .max(1.0);
        let slack = AFFINE_ULPS * f64::EPSILON * primary;
        check!(
            (via_kelvin.raw() - direct.raw()).abs() <= slack,
            "C -> F direct gave {} but C -> K -> F gave {} (slack {:e})",
            direct.raw(),
            via_kelvin.raw(),
            slack
        );
    }
}
