//! Type-level dimension markers and the [`Dimensioned`] bridge to the runtime [`Dimension`] enum.
//!
//! The [`Dimension`] enum and the dimension *traits* ([`Length`](super::Length),
//! [`Force`](super::Force), …) name dimensions at runtime and in trait position respectively.
//! The [`dim`] markers add the third form: zero-sized *type-level* tags usable as a generic
//! parameter or associated type, so a unit-of-measure protocol can enforce "same dimension" at
//! compile time (`convert::<To>()` only typechecks when `To::Dim == Self::Dim`).
//!
//! [`Dimensioned`] is the canonical bridge: it maps each marker to its runtime [`Dimension`] (and,
//! for free, its SI [`BaseExponents`]), so a consumer that declares a dimension twice — once as a
//! type-level marker, once as a runtime [`Quantity::DIMENSION`](super::Quantity::DIMENSION) — can
//! assert the two agree instead of letting them drift.

use super::{BaseExponents, Dimension};

/// The bridge from a type-level [`dim`] marker to its runtime [`Dimension`].
///
/// # Enforcing a dimension at compile time
///
/// Carry a marker as a type parameter and the compiler does the checking. Same dimension,
/// fine:
///
/// ```
/// use core::marker::PhantomData;
/// use euclid_units::{Dimensioned, dim};
///
/// #[derive(Copy, Clone)]
/// struct Measured<D: Dimensioned>(f64, PhantomData<D>);
///
/// impl<D: Dimensioned> Measured<D> {
///     fn new(v: f64) -> Self { Self(v, PhantomData) }
///     fn add(self, other: Self) -> Self { Self::new(self.0 + other.0) }
///     fn dimension(self) -> euclid_units::Dimension { D::DIMENSION }
/// }
///
/// let a = Measured::<dim::Length>::new(3.0);
/// let b = Measured::<dim::Length>::new(4.0);
/// assert_eq!(a.add(b).0, 7.0);
/// assert_eq!(a.dimension(), euclid_units::Dimension::Length);
/// ```
///
/// Mixing dimensions does not compile. A length plus a time is a type error, not a runtime
/// check — which is the entire reason the markers exist:
///
/// ```compile_fail,E0308
/// use core::marker::PhantomData;
/// use euclid_units::{Dimensioned, dim};
///
/// #[derive(Copy, Clone)]
/// struct Measured<D: Dimensioned>(f64, PhantomData<D>);
///
/// impl<D: Dimensioned> Measured<D> {
///     fn new(v: f64) -> Self { Self(v, PhantomData) }
///     fn add(self, other: Self) -> Self { Self::new(self.0 + other.0) }
/// }
///
/// let length = Measured::<dim::Length>::new(3.0);
/// let time = Measured::<dim::Time>::new(4.0);
/// let _ = length.add(time); // error: expected Measured<Length>, found Measured<Time>
/// ```
///
/// Nor does assigning across dimensions:
///
/// ```compile_fail,E0308
/// use core::marker::PhantomData;
/// use euclid_units::{Dimensioned, dim};
///
/// struct Measured<D: Dimensioned>(f64, PhantomData<D>);
///
/// let area: Measured<dim::Area> = Measured(1.0, PhantomData::<dim::Volume>);
/// ```
///
/// Commensurability is deliberately *not* enough. [`Energy`](super::Energy) and
/// [`Torque`](super::Torque) share the SI signature `[L² M T⁻²]`, so
/// [`Dimension::commensurable`] reports them compatible at runtime — but they are different
/// names, so their markers are different types and the compiler keeps them apart:
///
/// ```compile_fail,E0308
/// use core::marker::PhantomData;
/// use euclid_units::{Dimensioned, dim};
///
/// #[derive(Copy, Clone)]
/// struct Measured<D: Dimensioned>(f64, PhantomData<D>);
///
/// impl<D: Dimensioned> Measured<D> {
///     fn new(v: f64) -> Self { Self(v, PhantomData) }
///     fn add(self, other: Self) -> Self { Self::new(self.0 + other.0) }
/// }
///
/// let energy = Measured::<dim::Energy>::new(1.0);
/// let torque = Measured::<dim::Torque>::new(1.0);
/// let _ = energy.add(torque); // commensurable, but not the same name
/// ```
///
/// And a type that is not a marker cannot be used as one, so the protocol cannot be bypassed
/// by substituting an arbitrary type:
///
/// ```compile_fail,E0277
/// use core::marker::PhantomData;
/// use euclid_units::Dimensioned;
///
/// struct Measured<D: Dimensioned>(f64, PhantomData<D>);
///
/// // `f64` has no `Dimensioned` impl.
/// let _: Measured<f64> = Measured(1.0, PhantomData);
/// ```
pub trait Dimensioned: Copy + 'static {
    /// The runtime dimension this marker stands for.
    const DIMENSION: Dimension;
    /// The SI base-dimension exponents of this marker, derived from [`DIMENSION`](Self::DIMENSION).
    const EXPONENTS: BaseExponents = Self::DIMENSION.exponents();
}

macro_rules! markers {
    ($($name:ident),+ $(,)?) => {
        /// Zero-sized type-level markers, one per [`Dimension`] variant. Two units are
        /// inter-convertible only when they share the same marker, which the compiler enforces.
        pub mod dim {
            $(
                #[doc = concat!("Type-level marker for the `", stringify!($name), "` dimension.")]
                #[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
                pub struct $name;
            )+
        }

        $(
            impl Dimensioned for dim::$name {
                const DIMENSION: Dimension = Dimension::$name;
            }
        )+
    };
}

markers!(
    // base
    Length,
    Mass,
    Time,
    ElectricCurrent,
    Temperature,
    AmountOfSubstance,
    LuminousIntensity,
    // geometric
    Area,
    Volume,
    Angle,
    SolidAngle,
    // kinematic
    Frequency,
    Speed,
    Acceleration,
    Jerk,
    AngularVelocity,
    AngularAcceleration,
    VolumetricFlowRate,
    // mechanical
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
    // electromagnetic
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
    // thermal
    HeatCapacity,
    ThermalConductivity,
    SpecificHeatCapacity,
    ThermalResistance,
    HeatFlux,
    // photometric
    LuminousFlux,
    Illuminance,
    Luminance,
    // chemical
    CatalyticActivity,
    Concentration,
    MolarMass,
    MolarEnergy,
    MolarVolume,
    Molality,
    // radiological
    Radioactivity,
    AbsorbedDose,
    DoseEquivalent,
    // information
    Information,
    // pure number
    Dimensionless,
);
