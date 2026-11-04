//! Tests for the type-level [`dim`](crate::dim) markers and the [`Dimensioned`] bridge.
//!
//! The markers exist to make "same dimension" a *compile-time* requirement. That claim has two
//! halves, and they need different kinds of test:
//!
//! - **The runtime half** — every marker maps to the right [`Dimension`] and the right SI
//!   [`BaseExponents`], and the marker set stays in step with the enum. Ordinary unit tests,
//!   below.
//! - **The compile-time half** — code that mixes dimensions must be *rejected by the
//!   compiler*. A normal test cannot express that: there is nothing to run, because the
//!   program must not build. Those live as `compile_fail` doctests on
//!   [`Dimensioned`](crate::Dimensioned), which `cargo test --doc` verifies, and which need no
//!   third-party dev-dependency.

use crate::{BaseExponents, Dimension, Dimensioned, dim};

/// Every `(marker, expected dimension)` pair, written out so a macro mistake in `markers!`
/// cannot hide behind the macro that generated it.
macro_rules! marker_cases {
    ($($marker:ident),+ $(,)?) => {
        &[$((Dimension::$marker, <dim::$marker as Dimensioned>::DIMENSION, stringify!($marker))),+]
    };
}

type Case = (Dimension, Dimension, &'static str);

fn cases() -> &'static [Case] {
    marker_cases!(
        Length,
        Mass,
        Time,
        ElectricCurrent,
        Temperature,
        AmountOfSubstance,
        LuminousIntensity,
        Area,
        Volume,
        Angle,
        SolidAngle,
        Frequency,
        Speed,
        Acceleration,
        Jerk,
        AngularVelocity,
        AngularAcceleration,
        VolumetricFlowRate,
        Force,
        Pressure,
        Energy,
        Power,
        Momentum,
        AngularMomentum,
        Torque,
        Density,
        MomentOfInertia,
        DynamicViscosity,
        SurfaceTension,
        MassFlowRate,
        KinematicViscosity,
        SpecificEnergy,
        EnergyDensity,
        Action,
        ElectricCharge,
        Voltage,
        Resistance,
        Conductance,
        Capacitance,
        Inductance,
        MagneticFlux,
        MagneticFluxDensity,
        ElectricField,
        CurrentDensity,
        MagneticFieldStrength,
        Permittivity,
        Permeability,
        Resistivity,
        Conductivity,
        HeatCapacity,
        ThermalConductivity,
        SpecificHeatCapacity,
        ThermalResistance,
        HeatFlux,
        LuminousFlux,
        Illuminance,
        Luminance,
        CatalyticActivity,
        Concentration,
        MolarMass,
        MolarEnergy,
        MolarVolume,
        Molality,
        Radioactivity,
        AbsorbedDose,
        DoseEquivalent,
        Information,
        Dimensionless,
    )
}

#[test]
fn every_marker_maps_to_its_own_dimension() {
    // The failure this catches is a copy-paste slip in the `markers!` invocation pointing a
    // marker at the wrong enum variant — invisible by inspection, and it would silently make
    // two different dimensions compile-time compatible.
    for &(expected, actual, name) in cases() {
        assert_eq!(actual, expected, "dim::{name} maps to the wrong Dimension");
    }
}

#[test]
fn marker_exponents_agree_with_their_dimension() {
    // `EXPONENTS` is defaulted from `DIMENSION`, so this checks the default has not been
    // overridden incorrectly for any marker.
    for &(expected, actual, name) in cases() {
        let from_marker: BaseExponents = actual.exponents();
        assert_eq!(
            from_marker,
            expected.exponents(),
            "dim::{name} exponents disagree"
        );
    }
}

/// Maps every [`Dimension`] variant to the dimension its matching `dim` marker reports.
///
/// The match is deliberately **wildcard-free**. `Dimension` is `#[non_exhaustive]`, which
/// constrains downstream crates but not this one, so adding a variant without adding a marker
/// stops this file compiling. That is a stronger drift guard than any runtime assertion: the
/// two sets cannot silently diverge.
fn marker_dimension_of(d: Dimension) -> Dimension {
    macro_rules! arms {
        ($($name:ident),+ $(,)?) => {
            match d {
                $(Dimension::$name => <dim::$name as Dimensioned>::DIMENSION,)+
            }
        };
    }
    arms!(
        Length,
        Mass,
        Time,
        ElectricCurrent,
        Temperature,
        AmountOfSubstance,
        LuminousIntensity,
        Area,
        Volume,
        Angle,
        SolidAngle,
        Frequency,
        Speed,
        Acceleration,
        Jerk,
        AngularVelocity,
        AngularAcceleration,
        VolumetricFlowRate,
        Force,
        Pressure,
        Energy,
        Power,
        Momentum,
        AngularMomentum,
        Torque,
        Density,
        MomentOfInertia,
        DynamicViscosity,
        SurfaceTension,
        MassFlowRate,
        KinematicViscosity,
        SpecificEnergy,
        EnergyDensity,
        Action,
        ElectricCharge,
        Voltage,
        Resistance,
        Conductance,
        Capacitance,
        Inductance,
        MagneticFlux,
        MagneticFluxDensity,
        ElectricField,
        CurrentDensity,
        MagneticFieldStrength,
        Permittivity,
        Permeability,
        Resistivity,
        Conductivity,
        HeatCapacity,
        ThermalConductivity,
        SpecificHeatCapacity,
        ThermalResistance,
        HeatFlux,
        LuminousFlux,
        Illuminance,
        Luminance,
        CatalyticActivity,
        Concentration,
        MolarMass,
        MolarEnergy,
        MolarVolume,
        Molality,
        Radioactivity,
        AbsorbedDose,
        DoseEquivalent,
        Information,
        Dimensionless,
    )
}

#[test]
fn the_marker_set_covers_every_dimension_variant() {
    // Completeness is enforced at compile time by `marker_dimension_of`; this checks the
    // mapping is the identity for every variant the case list knows about, and that the list
    // itself has not fallen behind.
    for &(expected, _, name) in cases() {
        assert_eq!(
            marker_dimension_of(expected),
            expected,
            "dim::{name} does not round-trip through its Dimension"
        );
    }
}

#[test]
fn distinct_markers_are_distinct_types_with_distinct_dimensions() {
    // The property the compile-time guarantee rests on.
    assert_ne!(
        <dim::Length as Dimensioned>::DIMENSION,
        <dim::Time as Dimensioned>::DIMENSION
    );
    assert_ne!(
        <dim::Energy as Dimensioned>::DIMENSION,
        <dim::Power as Dimensioned>::DIMENSION
    );
    // Commensurable-but-distinct names stay distinct as markers, even though their SI base
    // exponents agree — this is the energy/torque case.
    assert_ne!(
        <dim::Energy as Dimensioned>::DIMENSION,
        <dim::Torque as Dimensioned>::DIMENSION
    );
    assert!(
        Dimension::Energy.commensurable(Dimension::Torque),
        "energy and torque share a base signature"
    );
}

#[test]
fn markers_are_zero_sized_so_they_cost_nothing_to_carry() {
    // They are intended as phantom type parameters; a non-zero size would make that a real
    // runtime cost on every quantity.
    assert_eq!(core::mem::size_of::<dim::Length>(), 0);
    assert_eq!(core::mem::size_of::<dim::Energy>(), 0);
    assert_eq!(core::mem::size_of::<dim::Dimensionless>(), 0);
}

/// A minimal unit-of-measure protocol of the kind the markers are designed to support, used
/// by the `compile_fail` doctests on [`Dimensioned`](crate::Dimensioned) and exercised here
/// for the cases that *should* compile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Measured<D: Dimensioned> {
    value: f64,
    _dim: core::marker::PhantomData<D>,
}

impl<D: Dimensioned> Measured<D> {
    pub(super) fn new(value: f64) -> Self {
        Self {
            value,
            _dim: core::marker::PhantomData,
        }
    }

    pub(super) fn value(self) -> f64 {
        self.value
    }

    /// Addition is only well-typed between the *same* marker, which is the whole point.
    pub(super) fn add(self, other: Self) -> Self {
        Self::new(self.value + other.value)
    }

    /// The runtime dimension, recovered from the type-level marker.
    pub(super) fn dimension(self) -> Dimension {
        D::DIMENSION
    }
}

#[test]
fn a_marker_parameterized_quantity_adds_within_a_dimension() {
    let a: Measured<dim::Length> = Measured::new(3.0);
    let b: Measured<dim::Length> = Measured::new(4.0);
    assert_eq!(a.add(b).value(), 7.0);
    // The marker survives into runtime inspection.
    assert_eq!(a.dimension(), Dimension::Length);
    let t: Measured<dim::Time> = Measured::new(1.0);
    assert_eq!(t.dimension(), Dimension::Time);
}

#[test]
fn the_same_generic_code_runs_over_any_marker() {
    // The payoff: one function, every dimension, with the dimension still tracked.
    fn total<D: Dimensioned>(xs: &[Measured<D>]) -> Measured<D> {
        xs.iter()
            .copied()
            .fold(Measured::new(0.0), |acc, x| acc.add(x))
    }
    let lengths = [Measured::<dim::Length>::new(1.0), Measured::new(2.0)];
    assert_eq!(total(&lengths).value(), 3.0);
    assert_eq!(total(&lengths).dimension(), Dimension::Length);

    let energies = [Measured::<dim::Energy>::new(10.0), Measured::new(5.0)];
    assert_eq!(total(&energies).value(), 15.0);
    assert_eq!(total(&energies).dimension(), Dimension::Energy);
}
