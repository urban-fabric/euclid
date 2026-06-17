use crate::{MassGrams, MassKilograms, MassMilligrams, Quantity};

#[test]
fn mass_round_trips() {
    let kg = MassKilograms::new(2.5);
    assert_eq!(kg.to_grams(), MassGrams::new(2500.0));
    assert_eq!(kg.to_milligrams(), MassMilligrams::new(2_500_000.0));
    assert_eq!(MassGrams::new(2500.0).to_kilograms(), kg);
    assert_eq!(MassMilligrams::new(2_500_000.0).to_kilograms(), kg);
}

#[test]
fn gram_and_milligram_conversions() {
    assert_eq!(
        MassGrams::new(1.25).to_milligrams(),
        MassMilligrams::new(1250.0)
    );
    assert_eq!(MassMilligrams::new(1250.0).to_grams(), MassGrams::new(1.25));
}

#[test]
fn mass_quantity_impls() {
    assert_eq!(MassKilograms::DIMENSION, crate::Dimension::Mass);
    assert_eq!(MassKilograms::UNIT_NAME, "kg");
    assert_eq!(MassGrams::UNIT_NAME, "g");
    assert_eq!(MassMilligrams::UNIT_NAME, "mg");
}
