mod length_feet;
mod length_kilometers;
mod length_meters;

pub use length_feet::LengthFeet;
pub use length_kilometers::LengthKm;
pub use length_meters::LengthMeters;

const M_PER_KM: f64 = 1_000.0;
const M_PER_FOOT: f64 = 0.304_8;
