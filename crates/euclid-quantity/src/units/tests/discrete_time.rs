//! Integer-backed durations and signed deltas, anchored to the SI time prefixes.

use std::time::Duration;

use euclid_units::Measure;

use crate::{
    TimeMicros, TimeMicrosDelta, TimeMillis, TimeNanos, TimeNanosDelta, TimePicos, TimePicosDelta,
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

    // Check raw counts for the nanosecond and picosecond resolutions too.
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
