//! The lazy unit newtypes. Each macro line is the lazy peer of a concrete `oresme`
//! unit; adding one is a single line.

lazy_linear_unit!(
    /// Length in meters.
    LazyMeters,
    crate::units::Dimension::Length,
    "m"
);

lazy_linear_unit!(
    /// Length in kilometers.
    LazyKm,
    crate::units::Dimension::Length,
    "km"
);

lazy_linear_unit!(
    /// Time in seconds.
    LazySeconds,
    crate::units::Dimension::Time,
    "s"
);

lazy_linear_unit!(
    /// Speed in meters per second.
    LazySpeedMs,
    crate::units::Dimension::Speed,
    "m/s"
);

lazy_linear_unit!(
    /// Area in square meters.
    LazySquareMeters,
    crate::units::Dimension::Area,
    "m²"
);

/// Meters per kilometer.
const M_PER_KM: f64 = 1_000.0;

impl<T: crate::LazyReal> LazyKm<T> {
    /// Convert to meters, lifting the `1000` factor into the carrier.
    pub fn to_meters(self) -> LazyMeters<T> {
        let factor = self.0.scalar_like(M_PER_KM);
        LazyMeters(self.0 * factor)
    }
}
