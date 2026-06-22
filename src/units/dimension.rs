/// The physical dimension a quantity measures.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Dimension {
    Acceleration,
    Area,
    Length,
    Mass,
    Temperature,
    Time,
    Speed,
    Angle,
    Dimensionless,
}

impl Dimension {
    /// Const-evaluable equality. The derived `PartialEq` cannot run in `const` context,
    /// where a unit's [`Quantity`](crate::units::Quantity) impl can assert that its runtime
    /// dimension matches its type-level marker.
    pub const fn same(self, other: Self) -> bool {
        self as u8 == other as u8
    }
}
