pub mod affine;
pub mod interval;

pub use affine::Point;
pub use interval::Interval;

#[cfg(test)]
mod arithmetic_tests;
