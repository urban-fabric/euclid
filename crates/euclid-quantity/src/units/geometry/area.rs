mod area_square_kilometers;
mod area_square_meters;

pub use area_square_kilometers::AreaSquareKilometers;
pub use area_square_meters::AreaSquareMeters;

const SQUARE_METERS_PER_SQUARE_KILOMETER: f64 = 1_000_000.0;
