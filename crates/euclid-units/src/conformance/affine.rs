//! The affine units, where the additive laws do **not** apply.
//!
//! Temperature is the reason this is a separate contract rather than a stricter
//! [`Measure`](crate::Measure):
//! `20°C + 20°C` is not `40°C`, because the scale has an arbitrary zero. What survives is the
//! *difference* — a delta is a proper measure even where the absolute value is not — so these
//! laws check the absolute/delta split rather than linearity.

/// The **temperature scales**, pinned to physical fixed points.
///
/// [`Temperature`](crate::Temperature) is the one tier in this crate whose conversions are
/// *affine* rather than linear -- each scale has both a factor and an offset -- and that is
/// exactly the arithmetic
/// that goes wrong quietly. A dropped offset still round-trips within one scale; a factor applied
/// to the offset still gives plausible-looking numbers; a swapped multiply and divide is only
/// visible away from the reference point. Mutation testing found twenty-one survivors across the
/// three scales, including every `from_*` constructor replaceable with a default and every
/// accessor with a constant.
///
/// # Fixed points, not round trips
///
/// A round trip through one scale cannot see a wrong offset, because the offset cancels. So the
/// anchors here are **physical constants**: water's freezing and boiling points and absolute
/// zero, in all three scales. They come from outside the implementation, which is what makes
/// them worth asserting -- the same reasoning as checking the SI exponent table against
/// definitions rather than against a copy of itself.
///
/// `-40` earns its place: it is the single temperature at which Celsius and Fahrenheit agree, so
/// an implementation that has confused the factor with the offset passes a `0` check and fails
/// here.
///
/// The tolerance is absolute and tight. These are small numbers reached by one multiply and one
/// add, so there is no accumulation to accommodate; `1e-9` is generous and still four orders
/// inside what a mistaken constant would cost.
pub fn temperature_laws<Q>(what: &str)
where
    Q: crate::Temperature<Repr = f64> + Copy + core::fmt::Debug,
{
    const TOL: f64 = 1e-9;

    let close = |got: f64, want: f64, law: &str| {
        assert!(
            (got - want).abs() <= TOL,
            "{what}: {law} — got {got}, want {want}"
        );
    };

    // Each scale reads back what it was given: the weakest law, and the only one a dropped
    // offset survives.
    for v in [0.0_f64, 25.0, -40.0, 100.0] {
        close(Q::from_celsius(v).celsius(), v, "from_celsius then celsius");
        close(
            Q::from_fahrenheit(v).fahrenheit(),
            v,
            "from_fahrenheit then fahrenheit",
        );
    }
    for v in [0.0_f64, 273.15, 373.15] {
        close(Q::from_kelvin(v).kelvin(), v, "from_kelvin then kelvin");
    }

    // Water freezes: 0 °C = 32 °F = 273.15 K.
    let freezing = Q::from_celsius(0.0);
    close(freezing.fahrenheit(), 32.0, "water freezes at 32 °F");
    close(freezing.kelvin(), 273.15, "water freezes at 273.15 K");

    // Water boils: 100 °C = 212 °F = 373.15 K.
    let boiling = Q::from_celsius(100.0);
    close(boiling.fahrenheit(), 212.0, "water boils at 212 °F");
    close(boiling.kelvin(), 373.15, "water boils at 373.15 K");

    // The crossover: -40 °C = -40 °F. An implementation that has swapped the factor and the
    // offset agrees with the others at zero and disagrees here.
    let crossover = Q::from_celsius(-40.0);
    close(
        crossover.fahrenheit(),
        -40.0,
        "-40 °C is -40 °F (the scales cross)",
    );
    close(crossover.kelvin(), 233.15, "-40 °C is 233.15 K");

    // Absolute zero, from every direction.
    let zero = Q::absolute_zero();
    close(zero.kelvin(), 0.0, "absolute zero is 0 K");
    close(zero.celsius(), -273.15, "absolute zero is -273.15 °C");
    close(zero.fahrenheit(), -459.67, "absolute zero is -459.67 °F");
    close(Q::from_kelvin(0.0).celsius(), -273.15, "0 K is -273.15 °C");
    close(
        Q::from_fahrenheit(-459.67).kelvin(),
        0.0,
        "-459.67 °F is 0 K",
    );

    // And every constructor/accessor pair agrees about the same physical temperature, which is
    // what makes the three scales one quantity rather than three.
    for celsius in [-40.0_f64, 0.0, 21.0, 100.0] {
        let from_c = Q::from_celsius(celsius);
        let from_f = Q::from_fahrenheit(celsius * 9.0 / 5.0 + 32.0);
        let from_k = Q::from_kelvin(celsius + 273.15);
        close(from_f.celsius(), celsius, "Fahrenheit agrees with Celsius");
        close(from_k.celsius(), celsius, "Kelvin agrees with Celsius");
        close(
            from_c.kelvin(),
            from_k.kelvin(),
            "the three constructors agree",
        );
    }
}
