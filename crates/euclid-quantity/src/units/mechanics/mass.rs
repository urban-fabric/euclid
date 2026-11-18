mod mass_grams;
mod mass_kilograms;
mod mass_milligrams;

pub use mass_grams::MassGrams;
pub use mass_kilograms::MassKilograms;
pub use mass_milligrams::MassMilligrams;

const KG_PER_GRAM: f64 = 0.001;
const KG_PER_MILLIGRAM: f64 = 0.000_001;
