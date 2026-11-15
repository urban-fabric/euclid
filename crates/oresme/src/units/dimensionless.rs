//! Dimensionless ratios: scaling factors and percentages.

mod factor;
mod percentage;

pub use factor::*;
pub use percentage::*;

#[cfg(test)]
mod factor_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use crate::*;

    #[test]
    fn factor_constants() {
        assert_eq!(Factor::IDENTITY.as_f64(), 1.0);
        assert_eq!(Factor::ZERO.as_f64(), 0.0);
        assert_eq!(Factor::DIMENSION, Dimension::Dimensionless);
        assert_eq!(Factor::UNIT_NAME, "");
    }

    #[test]
    fn factor_default_is_identity() {
        assert_eq!(Factor::default(), Factor::IDENTITY);
    }

    #[test]
    fn factor_composes() {
        let a = Factor::new(0.5);
        let b = Factor::new(2.0);
        assert_eq!((a * b).as_f64(), 1.0);
        assert_eq!((a * Factor::IDENTITY).as_f64(), 0.5);
        assert_eq!((Factor::IDENTITY * a).as_f64(), 0.5);
        assert_eq!((b / a).as_f64(), 4.0);
    }

    #[test]
    fn factor_scalar() {
        let a = Factor::new(0.5);
        assert_eq!((a * 4.0).as_f64(), 2.0);
        assert_eq!((2.0 * a).as_f64(), 1.0);
        assert_eq!((a + a).as_f64(), 1.0);
    }
}
#[cfg(test)]
mod percentage_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use crate::*;

    #[test]
    fn constants() {
        assert_eq!(Percentage::ZERO.as_percent(), 0.0);
        assert_eq!(Percentage::FULL.as_percent(), 100.0);
        assert_eq!(Percentage::DIMENSION, Dimension::Dimensionless);
        assert_eq!(Percentage::UNIT_NAME, "%");
    }

    #[test]
    fn fraction_round_trip() {
        let p = Percentage::from_fraction(Factor::new(0.42));
        assert!((p.as_percent() - 42.0).abs() < 1e-12);
        let f = p.to_fraction();
        assert!((f.as_f64() - 0.42).abs() < 1e-12);
    }

    #[test]
    fn predicates() {
        assert!(Percentage::ZERO.is_zero());
        assert!(Percentage::ZERO.is_non_negative());
        assert!(!Percentage::ZERO.is_positive());
        assert!(Percentage::FULL.is_in_range());
        assert!(Percentage::new(50.0).is_in_range());
        assert!(!Percentage::new(101.0).is_in_range());
        assert!(!Percentage::new(-1.0).is_in_range());
    }
}
