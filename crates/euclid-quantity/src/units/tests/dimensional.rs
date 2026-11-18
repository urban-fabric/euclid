//! Cross-dimension arithmetic by receiver family. Products and quotients of 3 and 2 are
//! exact. The method name supplies the operation, and euclid-units tests the dimension
//! exponents separately against the physical definitions.

/// `times_x` must multiply the magnitudes.
macro_rules! times_method {
    ($unit:ty, $tr:path, $method:ident, $arg:ident, $ret:ident) => {{
        let a = <$unit as euclid_units::Measure>::from_raw(3.0);
        let b = <<$unit as $tr>::$arg as euclid_units::Measure>::from_raw(2.0);
        assert_eq!(
            <$unit as $tr>::$method(a, b),
            <<$unit as $tr>::$ret as euclid_units::Measure>::from_raw(6.0),
            concat!(
                stringify!($unit),
                "::",
                stringify!($method),
                " must multiply the magnitudes (3 x 2 = 6)"
            )
        );
    }};
}

/// `per_x` must divide the magnitudes.
macro_rules! per_method {
    ($unit:ty, $tr:path, $method:ident, $arg:ident, $ret:ident) => {{
        let a = <$unit as euclid_units::Measure>::from_raw(3.0);
        let b = <<$unit as $tr>::$arg as euclid_units::Measure>::from_raw(2.0);
        assert_eq!(
            <$unit as $tr>::$method(a, b),
            <<$unit as $tr>::$ret as euclid_units::Measure>::from_raw(1.5),
            concat!(
                stringify!($unit),
                "::",
                stringify!($method),
                " must divide the magnitudes (3 / 2 = 1.5)"
            )
        );
    }};
}

/// `squared` and `cubed` must be the repeated product.
macro_rules! power_method {
    ($unit:ty, $tr:path, $method:ident, $ret:ident, $want:expr) => {{
        let a = <$unit as euclid_units::Measure>::from_raw(3.0);
        assert_eq!(
            <$unit as $tr>::$method(a),
            <<$unit as $tr>::$ret as euclid_units::Measure>::from_raw($want),
            concat!(
                stringify!($unit),
                "::",
                stringify!($method),
                " must be the repeated product"
            )
        );
    }};
}

mod chemistry;
mod electromagnetism;
mod geometry;
mod kinematics;
mod mechanics;
mod thermal;
