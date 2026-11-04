//! `Dense`'s scalar surface: operators, `Element`, the mask/`Compare` surface, and the
//! full `Real` implementation applied elementwise.
//!
//! This is what lets `Dense` reach the capability-extension methods bounded
//! `where Self: Real` — on those, the tensor itself must satisfy the whole real contract.

use core::ops::{Add, Div, Mul, Neg, Sub};

use euclid_numeric::{Compare, Element, NormedDivisionAlgebra, Real};

use super::Dense;

// --- operators ----------------------------------------------------------

macro_rules! binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for Dense {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                self.zip(&rhs, |x, y| x $op y)
            }
        }
    };
}

binop!(Add, add, +);
binop!(Sub, sub, -);
binop!(Mul, mul, *);
binop!(Div, div, /);

impl Neg for Dense {
    type Output = Self;
    fn neg(self) -> Self {
        self.map(|x| -x)
    }
}

impl Element for Dense {
    fn zero() -> Self {
        Self::point(0.0)
    }
    fn one() -> Self {
        Self::point(1.0)
    }
}

// --- masks and comparison ------------------------------------------------

/// Masks are `0.0`/`1.0` tensors, so this carrier is single-dtype like a numeric array
/// backend: `Mask == Self`.
impl euclid_numeric::Boolean for Dense {
    fn and(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x != 0.0 && y != 0.0))
    }
    fn or(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x != 0.0 || y != 0.0))
    }
    fn xor(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from((x != 0.0) != (y != 0.0)))
    }
    fn not(&self) -> Self {
        self.map(|x| f64::from(x == 0.0))
    }
}

impl Compare for Dense {
    type Mask = Self;

    fn lt(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x < y))
    }
    fn le(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x <= y))
    }
    fn gt(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x > y))
    }
    fn ge(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x >= y))
    }
    fn eq(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x == y))
    }
    fn ne(&self, other: &Self) -> Self {
        self.zip(other, |x, y| f64::from(x != y))
    }

    fn select(mask: &Self, on_true: Self, on_false: Self) -> Self {
        let shape = Self::broadcast_shape([&on_true, &on_false, mask]);
        let n = shape.iter().product::<usize>().max(1);
        let data = (0..n)
            .map(|i| {
                if mask.lane(i) != 0.0 {
                    on_true.lane(i)
                } else {
                    on_false.lane(i)
                }
            })
            .collect();
        Self { shape, data }
    }

    /// A host loop: iterate while every lane of the predicate mask is true.
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> Self::Mask,
        B: Fn(C) -> C,
    {
        let mut state = init;
        let mut guard = 0u32;
        while cond(&state).data.iter().all(|&m| m != 0.0) {
            state = body(state);
            guard += 1;
            if guard > 1_000_000 {
                break;
            }
        }
        state
    }
}

// --- the real scalar surface, applied elementwise ------------------------

/// `Semiring` and `Ring` are blanket-implemented over their operators, but `Arithmetic` is
/// not, so a division-bearing carrier opts in explicitly.
impl euclid_numeric::Arithmetic for Dense {}

impl NormedDivisionAlgebra for Dense {
    type RealPart = Self;

    fn conjugate(&self) -> Self {
        self.clone()
    }
    fn norm_sqr(&self) -> Self {
        self.map(|x| x * x)
    }
    fn real_part(&self) -> Self {
        self.clone()
    }
    fn from_real(re: Self) -> Self {
        re
    }
    fn magnitude(&self) -> Self {
        self.map(f64::abs)
    }
}

/// Generate the `Real` methods, each delegating to `f64`'s implementation elementwise.
macro_rules! real_const {
    ($($name:ident),* $(,)?) => {
        $(fn $name() -> Self { Self::point(<f64 as Real>::$name()) })*
    };
}
macro_rules! real_unary_ref {
    ($($name:ident),* $(,)?) => {
        $(fn $name(&self) -> Self { self.map(|x| <f64 as Real>::$name(&x)) })*
    };
}
macro_rules! real_unary_own {
    ($($name:ident),* $(,)?) => {
        $(fn $name(self) -> Self { self.map(|x| <f64 as Real>::$name(x)) })*
    };
}
macro_rules! real_binary_own {
    ($($name:ident),* $(,)?) => {
        $(fn $name(self, other: Self) -> Self {
            self.zip(&other, |x, y| <f64 as Real>::$name(x, y))
        })*
    };
}

impl Real for Dense {
    type Integer = i64;

    fn scalar(x: f64) -> Self {
        Self::point(x)
    }

    /// Same-shape fill, so a lifted literal keeps the operand's shape context.
    fn scalar_like(&self, x: f64) -> Self {
        Self {
            shape: self.shape.clone(),
            data: vec![x; self.count()],
        }
    }

    real_const!(
        two,
        half,
        pi,
        tau,
        frac_pi_2,
        frac_pi_4,
        sqrt_2,
        frac_1_sqrt_2,
        ln_2,
        ln_pi,
        ln_tau,
        half_ln_tau,
        e,
        log2_e,
        log10_e,
        ln_10,
        log2_10,
        log10_2,
        euler_gamma,
        infinity,
        neg_infinity,
        nan,
    );

    fn is_nan(&self) -> Self {
        self.map(|x| f64::from(x.is_nan()))
    }
    fn is_infinite(&self) -> Self {
        self.map(|x| f64::from(x.is_infinite()))
    }
    fn is_finite(&self) -> Self {
        self.map(|x| f64::from(x.is_finite()))
    }

    real_unary_ref!(
        exp,
        ln,
        ln_1p,
        exp_m1,
        sqrt,
        abs,
        square,
        ln_1p_exp,
        ln_1m_exp,
        ln_1m,
        ln_exp_m1,
        sigmoid,
        ln_sigmoid,
        ln_1m_sigmoid,
        logit,
    );

    real_unary_own!(
        sin,
        cos,
        tan,
        asin,
        acos,
        atan,
        sinh,
        cosh,
        tanh,
        asinh,
        acosh,
        atanh,
        cbrt,
        recip,
        log2,
        floor,
        ceil,
        round,
        trunc,
        fract,
        signum,
        to_radians,
        to_degrees,
        erf,
        erfc,
        erf_inv,
        ln_gamma,
        digamma,
        trigamma,
        erfcx,
        bessel_i0e,
        bessel_i1e,
        gamma,
        ln_bessel_i0,
        bessel_i1_over_i0,
    );

    real_binary_own!(
        atan2, hypot, copysign, rem_euclid, min, max, gamma_lr, gamma_ur, xlogy, xlog1py, xlog1my,
        ln_add_exp, ln_sub_exp, ln_beta, beta,
    );

    fn powf(&self, p: Self) -> Self {
        self.zip(&p, |x, y| <f64 as Real>::powf(&x, y))
    }

    fn sin_cos(self) -> (Self, Self) {
        (Real::sin(self.clone()), Real::cos(self))
    }

    fn powi(self, n: i64) -> Self {
        self.map(|x| <f64 as Real>::powi(x, n))
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        self.zip3(&a, &b, <f64 as Real>::mul_add)
    }

    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.zip3(&lo, &hi, <f64 as Real>::clamp)
    }

    fn beta_reg(self, a: Self, b: Self) -> Self {
        self.zip3(&a, &b, <f64 as Real>::beta_reg)
    }
}
