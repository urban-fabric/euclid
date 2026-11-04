# euclid-units

Physical-dimension contracts over any `Real` carrier — the units foundation for
the Urban Fabric projects. Part of [euclid](../..).

```toml
[dependencies]
euclid-units = "0.3"
euclid-numeric = "0.3"   # for the Real carrier bound
```

This crate ships dimension *contracts*, not concrete unit types. It does not
define a `Meters` or `Seconds` struct; a consumer defines its own unit carriers
and implements these traits for them, which is what lets several Urban Fabric
projects agree on what a length *is* without agreeing on how to store one.

## Trait Layering

Every dimension trait refines one of three carrier-generic base contracts:

| trait | adds |
|-------|------|
| `Quantity` | the raw carrier, `Dimension` tag, and unit name |
| `ScalarQuantity` | min/max/clamp/rounding and mask-returning predicates |
| `LinearQuantity` | displacement algebra: add/sub/neg/scale/lerp |

Absolute scalars such as `Temperature` deliberately stop at `ScalarQuantity` —
there is no meaningful sum of two temperatures.

The dimension traits then add the cross-dimensional algebra (`length × length =
area`, `length / time = speed`, `mass × acceleration = force`, `current ×
resistance = voltage`, …) through associated quantity types.

## Coverage

The seven SI base dimensions, all 22 SI named derived units, and a broad set of
further derived quantities, grouped by domain:

| module | quantities |
|--------|-----------|
| base | `Length`, `Mass`, `Time`, `ElectricCurrent`, `Temperature`, `AmountOfSubstance`, `LuminousIntensity` |
| `geometry` | `Area`, `Volume`, `Angle`, `SolidAngle` |
| `kinematics` | `Frequency`, `Speed`, `Acceleration`, `Jerk`, angular velocity/acceleration, `VolumetricFlowRate` |
| `mechanics` | `Force`, `Pressure`, `Energy`, `Power`, momentum, torque, density, viscosities, `SpecificEnergy`, `EnergyDensity`, `Action` |
| `electromagnetism` | charge, `Voltage`, `Resistance`, conductance, capacitance, inductance, flux/flux density, field strength, permittivity, permeability, resistivity, conductivity |
| `thermal` | `HeatCapacity`, `ThermalConductivity`, `SpecificHeatCapacity`, `ThermalResistance`, `HeatFlux` |
| `photometry` | `LuminousFlux`, `Illuminance`, `Luminance` |
| `chemistry` | `CatalyticActivity`, `Concentration`, molar mass/energy/volume, `Molality` |
| `radiology` | `Radioactivity`, `AbsorbedDose`, `DoseEquivalent` |
| pure number | `Dimensionless` |

Commensurable-but-distinct names (energy vs. torque; absorbed dose vs. dose
equivalent vs. specific energy) are tracked separately by name while sharing a
base-dimension signature.

## Naming Convention

Cross-dimensional operations are verbs suffixed by the *operand's* dimension:
`times_<operand>` is `self * operand`, `per_<operand>` is `self / operand`. Pure
powers get `squared`/`cubed`, and reciprocal pairs (`time ⇄ frequency`,
`resistance ⇄ conductance`) are named semantically. Same-unit ratios collapse to
the raw carrier through `LinearQuantity::ratio`, and a `Dimensionless` factor
scales a dimensioned quantity through `LinearQuantity::scale`.

## Type-Level Markers

Alongside the runtime `Dimension` enum, the `dim` module provides a zero-sized
*type-level* marker per dimension, and `Dimensioned` bridges each marker back to
its runtime `Dimension` and SI `BaseExponents`. A unit-of-measure protocol can
carry a `dim::*` marker as an associated type to make "same dimension" a
compile-time requirement.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2026 Raymond Mead.
