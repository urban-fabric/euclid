//! The log-domain family: log-sum-exp, the sigmoid/logit pair, and the absorbing-zero products.
//!
//! Part of the [`identities`](super) suite.
//!
//! These twenty-odd methods are the ones a probabilistic consumer actually calls, every one of
//! them is real logic rather than a `std` delegation, and **none of them had a generated
//! property** before this file.
//!
//! # Two of these laws are exact, which is rarer than it sounds
//!
//! Measured over 4,000-point sweeps before being written here, `ln_add_exp` is **bit-exactly
//! symmetric** and `ln_1m_sigmoid(x)` is **bit-exactly** `ln_sigmoid(−x)` across `[−700, 700]`.
//! Both are asserted at zero ulps, which is the strongest statement available and the one a
//! future refactor is most likely to break: either could become "accurate to a few ulps" without
//! any accuracy test noticing.
//!
//! `max(a,b) ≤ ln_add_exp(a,b) ≤ max(a,b) + ln 2` is exact too, and holds at *every* argument
//! including the infinities — a bound rather than a value, so nothing can round it away.
//!
//! # Three laws that are true in mathematics and false in `f64`
//!
//! The measurement rejected three identities this file was drafted with, and they are recorded
//! because the mistake is the same one each time: **an identity that subtracts two nearly-equal
//! quantities is not a test, it is a cancellation.**
//!
//! - `ln_sigmoid(x) = ln(sigmoid(x))` reaches a relative error of **0.49** at `x = 36.7`.
//!   `sigmoid` saturates to exactly `1.0` there, so `ln(sigmoid(x))` is `0` while the true value
//!   is about `−1.1e-16`. That saturation is the entire reason `ln_sigmoid` exists, so asserting
//!   the naive form would be asserting the bug.
//! - `ln_1p_exp(x) − x = ln_1p_exp(−x)` reaches a relative error of **1.0** at `x = 33.6`, where
//!   `ln_1p_exp(x)` has rounded to exactly `x` and the left side is exactly zero. Restated as a
//!   *sum* — `ln_1p_exp(x) = x + ln_1p_exp(−x)` — it is well conditioned for `x ≥ 0` and is
//!   asserted below.
//! - `logit(sigmoid(x)) = x` degrades from `2.2e-13` on `[−1,1]` to `1.1e-2` on `[−36,36]`,
//!   because `1 − sigmoid(x)` falls below `ulp(1)` and the round trip cannot recover what the
//!   representation discarded. Its bound carries `e^{|x|}` explicitly for that reason.
//!
//! # And one convention worth pinning because it is a choice
//!
//! `xlogy(0, y)` is `0` for **every** `y` — including `±∞` and `NaN`. The absorbing zero wins
//! over NaN propagation, which is defensible and is not the only defensible answer, so it is
//! asserted rather than left to be rediscovered.

use super::*;

#[path = "log_domain/products.rs"]
mod products;
#[path = "log_domain/sigmoid.rs"]
mod sigmoid;
#[path = "log_domain/sum_exp.rs"]
mod sum_exp;
