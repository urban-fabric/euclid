mod speed_kilometers_per_hour;
mod speed_meters_per_second;
mod speed_miles_per_hour;

pub use speed_kilometers_per_hour::SpeedKph;
pub use speed_meters_per_second::SpeedMs;
pub use speed_miles_per_hour::SpeedMph;

const MS_PER_KPH: f64 = 1_000.0 / 3_600.0;
const MS_PER_MPH: f64 = 0.447_04;
const KPH_PER_MPH: f64 = 1.609_344;
const MPH_PER_KPH: f64 = 1.0 / KPH_PER_MPH;
