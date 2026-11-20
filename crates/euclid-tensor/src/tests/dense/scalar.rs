//! `Dense`'s scalar surface: operators, `Element`, the mask/`Compare` surface, and the
//! full `Real` implementation applied elementwise.
//!
//! This is what lets `Dense` reach the capability-extension methods bounded
//! `where Self: Real` — on those, the tensor itself must satisfy the whole real contract.

use core::ops::{Add, Div, Mul, Neg, Sub};

use euclid_scalar::{Compare, Element, NormedDivisionAlgebra, Real};

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
impl euclid_scalar::Boolean for Dense {
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
impl euclid_scalar::Arithmetic for Dense {}

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

mod real;
