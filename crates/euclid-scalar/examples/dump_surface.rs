//! Evaluate one function of the public surface at arguments read from stdin, bit-exactly.
//!
//! The Rust half of the local differential-accuracy harness. It knows nothing about references:
//! it reads `f64` bit patterns as hex, applies the named function, and writes the result's bit
//! pattern back. The oracle is a separate Python process binding MPFR at 256 bits, which is
//! where it has to live — CONTRIBUTING keeps arbitrary-precision arithmetic out of this
//! workspace, on the grounds that an oracle module here would be a second implementation of the
//! thing under test.
//!
//! Hex in, hex out, so nothing is lost to decimal formatting on either side of the pipe. A
//! `NaN` result is written as its bit pattern like any other, which is what lets the driver
//! check the off-domain answers too.
//!
//! ```sh
//! echo 3ff0000000000000 | cargo run --release --example dump_surface -- gamma
//! ```

// The workspace denies `panic`, `expect_used` and `unwrap_used`, which is right for a numerics
// library whose callers cannot catch a panic across an FFI boundary. This is a developer harness
// driven by a local script, not library code: a bad function name or unreadable stdin is a usage
// error, and failing loudly on the spot is the correct behaviour rather than something to thread
// a `Result` through. Allowed deliberately and scoped to this file.
#![allow(clippy::panic, clippy::expect_used)]

use euclid_scalar::Real;
use std::io::{BufWriter, Read, Write};

/// One-argument functions of the real surface, by the name the driver passes.
///
/// Called as `Real::f(x)` rather than `x.f()`: several of these names are candidates for
/// addition to `std`, and the method form would silently change meaning the day one lands.
///
/// Matched rather than looked up in a map: an unknown name is a usage error worth failing on,
/// and the list is the harness's contract with its driver.
fn apply(name: &str, x: f64) -> f64 {
    match name {
        "gamma" => Real::gamma(x),
        "ln_gamma" => Real::ln_gamma(x),
        "digamma" => Real::digamma(x),
        "trigamma" => Real::trigamma(x),
        "rgamma" => Real::rgamma(x),
        "erf" => Real::erf(x),
        "erfc" => Real::erfc(x),
        "erfcx" => Real::erfcx(x),
        "erf_inv" => Real::erf_inv(x),
        "erfc_inv" => Real::erfc_inv(x),
        "bessel_j0" => Real::bessel_j0(x),
        "bessel_j1" => Real::bessel_j1(x),
        "bessel_y0" => Real::bessel_y0(x),
        "bessel_y1" => Real::bessel_y1(x),
        "bessel_i0" => Real::bessel_i0(x),
        "bessel_i1" => Real::bessel_i1(x),
        "bessel_k0" => Real::bessel_k0(x),
        "bessel_k1" => Real::bessel_k1(x),
        "bessel_i0e" => Real::bessel_i0e(x),
        "bessel_k0e" => Real::bessel_k0e(x),
        "airy_ai" => Real::airy_ai(x),
        "airy_ai_prime" => Real::airy_ai_prime(x),
        "airy_bi" => Real::airy_bi(x),
        "airy_bi_prime" => Real::airy_bi_prime(x),
        "zeta" => Real::zeta(x),
        "exp_int_e1" => Real::exp_int_e1(x),
        "exp_int_ei" => Real::exp_int_ei(x),
        "sin_int" => Real::sin_int(x),
        "cos_int" => Real::cos_int(x),
        "lambert_w" => Real::lambert_w(x),
        "ndtr" => Real::ndtr(x),
        "ndtri" => Real::ndtri(x),
        "log_ndtr" => Real::log_ndtr(x),
        "exprel" => Real::exprel(x),
        "sinc" => Real::sinc(x),
        "cos_m1" => Real::cos_m1(x),
        "ln_bessel_i0" => Real::ln_bessel_i0(x),
        "bessel_i1_over_i0" => Real::bessel_i1_over_i0(x),
        // The scaled Airy quartet and the scaled modified Bessels: their own routes, none of
        // which the sweep had ever reached.
        "airy_ai_e" => Real::airy_ai_e(x),
        "airy_ai_e_prime" => Real::airy_ai_e_prime(x),
        "airy_bi_e" => Real::airy_bi_e(x),
        "airy_bi_e_prime" => Real::airy_bi_e_prime(x),
        "bessel_i1e" => Real::bessel_i1e(x),
        "bessel_k1e" => Real::bessel_k1e(x),
        // Elementary and structural. Mostly delegation, which is exactly why they are worth a
        // sweep: a delegation that is *not* a pass-through is invisible without one.
        "exp2" => Real::exp2(x),
        "log2" => Real::log2(x),
        "log10" => Real::log10(x),
        "cbrt" => Real::cbrt(x),
        "recip" => Real::recip(x),
        "asin" => Real::asin(x),
        "acos" => Real::acos(x),
        "atan" => Real::atan(x),
        "sinh" => Real::sinh(x),
        "cosh" => Real::cosh(x),
        "tanh" => Real::tanh(x),
        "asinh" => Real::asinh(x),
        "acosh" => Real::acosh(x),
        "atanh" => Real::atanh(x),
        // The circular trio. Worth measuring despite being delegation: a large argument needs
        // exact reduction modulo `pi/2`, which is the classic place a pass-through stops being
        // one.
        "sin" => Real::sin(x),
        "cos" => Real::cos(x),
        "tan" => Real::tan(x),
        other => panic!("unknown function {other}"),
    }
}

/// Two-argument functions, read as a pair of hex words per line.
fn apply2(name: &str, a: f64, b: f64) -> f64 {
    match name {
        "gamma_lr" => Real::gamma_lr(a, b),
        "gamma_ur" => Real::gamma_ur(a, b),
        "ln_beta" => Real::ln_beta(a, b),
        "beta" => Real::beta(a, b),
        "ln_rising_factorial" => Real::ln_rising_factorial(a, b),
        "xlogy" => Real::xlogy(a, b),
        "ln_add_exp" => Real::ln_add_exp(a, b),
        "ln_sub_exp" => Real::ln_sub_exp(a, b),
        "xlog1py" => Real::xlog1py(a, b),
        "xlog1my" => Real::xlog1my(a, b),
        "hypot" => Real::hypot(a, b),
        "atan2" => Real::atan2(a, b),
        "gamma_lr_inv" => Real::gamma_lr_inv(a, b),
        // `order` is a `u32`, so the second word is read as a small integer rather than as a
        // float — the driver sends it as the bit pattern of the integer's value.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        "polygamma" => Real::polygamma(a, b as u32),
        other => panic!("unknown two-argument function {other}"),
    }
}

/// Three-argument functions, read as a triple of hex words per line.
///
/// `beta_reg` and its inverse are the only members of the public surface that need three, and
/// they are also the pair carrying the deepest loop in the crate — so the harness had to grow a
/// third slot before either could be measured at all.
fn apply3(name: &str, a: f64, b: f64, c: f64) -> f64 {
    match name {
        "beta_reg" => Real::beta_reg(a, b, c),
        "beta_reg_inv" => Real::beta_reg_inv(a, b, c),
        "hypot3" => Real::hypot3(a, b, c),
        other => panic!("unknown three-argument function {other}"),
    }
}

/// The single-precision surface, selected by an `f32:` prefix on the function name.
///
/// Reads and writes `f64` bit patterns like everything else — the argument is narrowed on the way
/// in and widened on the way out — so the driver needs no second encoding. Narrowing means the
/// sweep must pass arguments that are exactly representable in `f32` to be comparing like with
/// like, which is the driver's job.
fn apply_f32(name: &str, x: f64) -> f64 {
    let y = x as f32;
    let out = match name {
        "bessel_i0" => Real::bessel_i0(y),
        "bessel_i1" => Real::bessel_i1(y),
        "bessel_i0e" => Real::bessel_i0e(y),
        "bessel_k0" => Real::bessel_k0(y),
        "gamma" => Real::gamma(y),
        "rgamma" => Real::rgamma(y),
        "erf" => Real::erf(y),
        "erf_inv" => Real::erf_inv(y),
        "erfc" => Real::erfc(y),
        "bessel_j0" => Real::bessel_j0(y),
        "bessel_y0" => Real::bessel_y0(y),
        "airy_ai" => Real::airy_ai(y),
        "airy_bi" => Real::airy_bi(y),
        other => panic!("unknown single-precision function {other}"),
    };
    f64::from(out)
}

fn main() {
    let name = std::env::args()
        .nth(1)
        .expect("usage: dump_surface <function>");
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("stdin should be readable");
    let out = std::io::stdout();
    let mut out = BufWriter::new(out.lock());
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut words = line.split_whitespace();
        let first = u64::from_str_radix(words.next().expect("a hex word"), 16)
            .expect("a 16-digit hex word");
        let value = match words.next() {
            None => match name.strip_prefix("f32:") {
                Some(single) => apply_f32(single, f64::from_bits(first)),
                None => apply(&name, f64::from_bits(first)),
            },
            Some(second) => {
                let second =
                    f64::from_bits(u64::from_str_radix(second, 16).expect("a 16-digit hex word"));
                match words.next() {
                    None => apply2(&name, f64::from_bits(first), second),
                    Some(third) => apply3(
                        &name,
                        f64::from_bits(first),
                        second,
                        f64::from_bits(
                            u64::from_str_radix(third, 16).expect("a 16-digit hex word"),
                        ),
                    ),
                }
            }
        };
        writeln!(out, "{:016x}", value.to_bits()).expect("stdout should be writable");
    }
}
