//! The integer-backed time units, and the per-unit helpers the generated suites could not reach.
//!
//! # Why these were outside everything else
//!
//! `inherent_tests` and `conformance_tests` are generated over the **f64** units: the generator
//! keys on `as_<unit>() -> f64`, and these types return `u64` or `i64`. They sit on
//! `DiscreteQuantity` rather than the `ScalarQuantity`/`LinearQuantity` tiers, so neither the
//! inherent-surface macro nor `quantity_laws` applies to them.
//!
//! That left thirty-three surviving mutants concentrated here — the largest remaining cluster in
//! the crate once the f64 units were covered, and the last structural one. Every conversion
//! (`to_millis`, `to_seconds`, `from_duration`), every accessor, `TimeMicrosDelta::between`, and
//! the `Add`/`AddAssign`/`Sum` family could be replaced with a default or have its operator
//! swapped.
//!
//! # The anchors are the SI prefixes
//!
//! A conversion between time units is fixed by arithmetic nobody gets to choose: a second is a
//! thousand milliseconds, a million microseconds, a billion nanoseconds. Asserting against those
//! rather than against the implementation's own divisor is what makes this test independent of
//! the code — the same reasoning as the temperature fixed points and the SI exponent table.
//!
//! The exact values are deliberate too: `1_000_000` microseconds is `1.0` seconds with no
//! rounding, so a swapped multiply and divide moves the answer by twelve orders of magnitude
//! rather than by an ulp.

use std::time::Duration;

use euclid_units::Measure;

use crate::{
    TimeMicros, TimeMicrosDelta, TimeMillis, TimeNanos, TimeNanosDelta, TimePicos, TimePicosDelta,
    TimeSeconds,
};

#[test]
fn the_discrete_accessors_and_raw_conversions_round_trip() {
    for raw in [0_u64, 1, 1_000, 1_000_000, u64::MAX / 2] {
        let t = TimeMicros::new(raw);
        assert_eq!(t.as_micros(), raw, "as_micros must be the raw count");
        assert_eq!(Measure::raw(&t), raw, "Measure::raw must agree");
        assert_eq!(
            <TimeMicros as Measure>::from_raw(raw),
            t,
            "from_raw must invert raw"
        );
    }

    for raw in [0_i64, 1, -1, 1_000, -1_000_000, i64::MIN / 2] {
        let d = TimeMicrosDelta::new(raw);
        assert_eq!(d.as_micros(), raw, "a delta is signed");
        assert_eq!(Measure::raw(&d), raw);
        assert_eq!(<TimeMicrosDelta as Measure>::from_raw(raw), d);
    }

    // The other three resolutions, whose `from_raw` was equally unpinned.
    for raw in [0_u64, 1, 1_000_000_000] {
        assert_eq!(
            Measure::raw(&<TimeNanos as Measure>::from_raw(raw)),
            raw,
            "TimeNanos from_raw/raw"
        );
        assert_eq!(
            Measure::raw(&<TimePicos as Measure>::from_raw(raw)),
            raw,
            "TimePicos from_raw/raw"
        );
    }

    // And their signed deltas. A non-zero raw is what matters here: these types derive `Default`,
    // so a `from_raw` replaced by a default is *correct* at zero and wrong everywhere else — the
    // first version of this loop started at zero and could not tell the two apart.
    for raw in [1_i64, -1, 1_000, -1_000_000_000] {
        assert_eq!(
            Measure::raw(&<TimeNanosDelta as Measure>::from_raw(raw)),
            raw,
            "TimeNanosDelta from_raw/raw"
        );
        assert_eq!(
            Measure::raw(&<TimePicosDelta as Measure>::from_raw(raw)),
            raw,
            "TimePicosDelta from_raw/raw"
        );
    }
}

#[test]
fn the_conversions_match_the_si_prefixes() {
    const TOL: f64 = 1e-12;
    let close = |got: f64, want: f64, law: &str| {
        assert!((got - want).abs() <= TOL, "{law}: got {got}, want {want}");
    };

    // A second is a million microseconds.
    close(
        TimeMicros::new(1_000_000).to_seconds().as_seconds(),
        1.0,
        "1e6 us is 1 s",
    );
    close(
        TimeMicros::new(0).to_seconds().as_seconds(),
        0.0,
        "0 us is 0 s",
    );
    close(
        TimeMicros::new(2_500_000).to_seconds().as_seconds(),
        2.5,
        "2.5e6 us is 2.5 s",
    );

    // A millisecond is a thousand microseconds.
    close(
        TimeMicros::new(1_000).to_millis().as_millis(),
        1.0,
        "1e3 us is 1 ms",
    );
    close(
        TimeMicros::new(1_000_000).to_millis().as_millis(),
        1_000.0,
        "1e6 us is 1e3 ms",
    );

    // A second is a billion nanoseconds and a trillion picoseconds.
    close(
        TimeNanos::new(1_000_000_000).to_seconds().as_seconds(),
        1.0,
        "1e9 ns is 1 s",
    );
    close(
        TimePicos::new(1_000_000_000_000).to_seconds().as_seconds(),
        1.0,
        "1e12 ps is 1 s",
    );
    close(
        TimePicos::new(1_000_000_000_000).to_millis().as_millis(),
        1_000.0,
        "1e12 ps is 1e3 ms",
    );
}

#[test]
fn from_duration_reads_the_duration() {
    for micros in [0_u64, 1, 1_000, 1_234_567] {
        assert_eq!(
            TimeMicros::from_duration(Duration::from_micros(micros)).as_micros(),
            micros,
            "from_duration must read the microsecond count"
        );
    }

    // `TimeMillis` is f64-backed and scales from whole seconds, so it gets the prefix check
    // rather than an identity.
    const TOL: f64 = 1e-9;
    for (secs, millis) in [(0_u64, 0.0), (1, 1_000.0), (3, 3_000.0)] {
        let got = TimeMillis::from_duration(Duration::from_secs(secs)).as_millis();
        assert!(
            (got - millis).abs() <= TOL,
            "from_duration({secs} s) = {got} ms, want {millis}"
        );
    }
    assert!(
        (TimeMillis::from_duration(Duration::from_millis(1_500)).as_millis() - 1_500.0).abs()
            <= TOL,
        "a sub-second duration must keep its milliseconds"
    );
}

/// `between` is signed and ordered: `after - before`, negative when time went backwards.
///
/// The sign is the entire reason `TimeMicrosDelta` exists — the source notes the unsigned
/// duration "could not say so" — and the subtraction could have been an addition or a division
/// without anything noticing.
#[test]
fn a_delta_is_the_signed_difference_in_the_right_order() {
    let earlier = TimeMicros::new(1_000);
    let later = TimeMicros::new(3_500);

    assert_eq!(
        TimeMicrosDelta::between(earlier, later).as_micros(),
        2_500,
        "between(before, after) is after - before"
    );
    assert_eq!(
        TimeMicrosDelta::between(later, earlier).as_micros(),
        -2_500,
        "a backwards interval is negative, which is why the delta is signed"
    );
    assert_eq!(
        TimeMicrosDelta::between(earlier, earlier).as_micros(),
        0,
        "an instant has no duration"
    );
}

#[test]
fn discrete_addition_agrees_across_its_three_spellings() {
    let a = TimeMicros::new(1_000);
    let b = TimeMicros::new(250);

    assert_eq!((a + b).as_micros(), 1_250, "+ must add");

    let mut assigned = a;
    assigned += b;
    assert_eq!(assigned, a + b, "+= must agree with +");

    let summed: TimeMicros = [a, b, TimeMicros::new(4)].into_iter().sum();
    assert_eq!(summed.as_micros(), 1_254, "sum must fold +");
    let empty: TimeMicros = core::iter::empty().sum();
    assert_eq!(empty, TimeMicros::ZERO, "the empty sum is zero");

    // The same three for picoseconds, whose `+=` was separately unpinned.
    let mut p = TimePicos::new(7);
    p += TimePicos::new(3);
    assert_eq!(p.as_picos(), 10, "TimePicos += must add");

    // `saturating_add` clamps instead of wrapping, which is what lets the type be unsigned.
    assert_eq!(
        TimeMicros::new(u64::MAX).saturating_add(TimeMicros::new(1)),
        TimeMicros::new(u64::MAX),
        "saturating_add must clamp at the maximum"
    );
}

/// The remaining per-unit helpers on the f64 units: the `f32` narrowing accessors, the positivity
/// assertion, and the two predicates the generated suites do not know about.
#[test]
fn the_per_unit_helpers_agree_with_the_raw_value() {
    use crate::{LengthMeters, SpeedKph};

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

/// `MolarVolume::concentration` is a reciprocal, which is the one dimensional method whose name
/// does not say `per_` or `times_` — so the generated dimensional suite does not cover it.
#[test]
fn molar_volume_inverts_to_concentration() {
    use euclid_units::MolarVolume;

    use crate::MolarVolumeCubicMetersPerMole;

    const TOL: f64 = 1e-12;
    for v in [0.5_f64, 1.0, 2.0, 4.0] {
        let got = MolarVolumeCubicMetersPerMole::new(v)
            .concentration()
            .as_moles_per_cubic_meter();
        assert!(
            (got - 1.0 / v).abs() <= TOL,
            "concentration({v}) = {got}, want {}",
            1.0 / v
        );
    }
}
