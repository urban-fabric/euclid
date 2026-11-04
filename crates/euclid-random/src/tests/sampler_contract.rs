//! The sampler contracts. The load-bearing property is *purity* — the same key must give the
//! same draw — since that is what makes a key-based API reproducible. The moment statistics
//! are checked, it is only to confirm the reference implementation is not nonsense.

use super::toy_sampler::ToySampler;
use crate::{CountSampler, GammaSampler, RandomKey, RandomSampler};

type S = ToySampler;

/// Scalar backends take the empty shape, per the `RandomSampler` docs.
const SCALAR: &[usize] = &[];

/// Draw `n` values, one per key folded off a root — the canonical way to get a sample.
fn sample<T, F>(n: u64, draw: F) -> Vec<T>
where
    F: Fn(&u64) -> T,
{
    let root = S::key(0xFEED);
    (0..n).map(|i| draw(&S::fold_in(&root, i))).collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

// --- purity: the actual contract ----------------------------------------

#[test]
fn every_draw_is_a_pure_function_of_its_key() {
    let k = S::key(1234);
    assert_eq!(
        S::uniform(&k, SCALAR, -1.0, 1.0),
        S::uniform(&k, SCALAR, -1.0, 1.0)
    );
    assert_eq!(
        S::normal(&k, SCALAR, 0.0, 1.0),
        S::normal(&k, SCALAR, 0.0, 1.0)
    );
    assert_eq!(S::bernoulli(&k, SCALAR, 0.5), S::bernoulli(&k, SCALAR, 0.5));
    assert_eq!(
        S::randint(&k, SCALAR, 0, 100),
        S::randint(&k, SCALAR, 0, 100)
    );
    assert_eq!(
        S::standard_gamma(&k, SCALAR, 2.5),
        S::standard_gamma(&k, SCALAR, 2.5)
    );
    assert_eq!(S::poisson(&k, SCALAR, 4.0), S::poisson(&k, SCALAR, 4.0));
    assert_eq!(
        S::binomial(&k, SCALAR, 20.0, 0.3),
        S::binomial(&k, SCALAR, 20.0, 0.3)
    );
    assert_eq!(
        S::beta_binomial(&k, SCALAR, 20.0, 2.0, 3.0),
        S::beta_binomial(&k, SCALAR, 20.0, 2.0, 3.0)
    );
}

#[test]
fn distinct_keys_give_distinct_draws() {
    let a = S::key(1);
    let b = S::key(2);
    assert_ne!(
        S::uniform(&a, SCALAR, 0.0, 1.0),
        S::uniform(&b, SCALAR, 0.0, 1.0)
    );
    assert_ne!(
        S::normal(&a, SCALAR, 0.0, 1.0),
        S::normal(&b, SCALAR, 0.0, 1.0)
    );
}

#[test]
fn distributions_drawn_from_one_key_are_not_aliases() {
    // Reusing a key across call sites must not make `uniform` and `normal` the same number.
    let k = S::key(77);
    assert_ne!(
        S::uniform(&k, SCALAR, 0.0, 1.0),
        S::normal(&k, SCALAR, 0.0, 1.0)
    );
}

// --- distributional sanity ----------------------------------------------

#[test]
fn uniform_lands_in_range_with_the_right_moments() {
    let xs = sample(20_000, |k| S::uniform(k, SCALAR, -2.0, 6.0));
    assert!(xs.iter().all(|&x| (-2.0..6.0).contains(&x)));
    assert!((mean(&xs) - 2.0).abs() < 0.1, "mean was {}", mean(&xs));
    // Var = (b-a)^2 / 12 = 64/12 ≈ 5.333
    assert!((variance(&xs) - 64.0 / 12.0).abs() < 0.2);
}

#[test]
fn normal_has_the_right_moments() {
    let xs = sample(20_000, |k| S::normal(k, SCALAR, 3.0, 2.0));
    assert!((mean(&xs) - 3.0).abs() < 0.1, "mean was {}", mean(&xs));
    assert!(
        (variance(&xs) - 4.0).abs() < 0.25,
        "var was {}",
        variance(&xs)
    );
}

#[test]
fn bernoulli_matches_its_probability() {
    let hits = sample(20_000, |k| S::bernoulli(k, SCALAR, 0.25))
        .into_iter()
        .filter(|&b| b)
        .count();
    let rate = hits as f64 / 20_000.0;
    assert!((rate - 0.25).abs() < 0.02, "rate was {rate}");
}

#[test]
fn bernoulli_is_degenerate_at_the_endpoints() {
    let never = sample(500, |k| S::bernoulli(k, SCALAR, 0.0));
    let always = sample(500, |k| S::bernoulli(k, SCALAR, 1.0));
    assert!(never.iter().all(|&b| !b));
    assert!(always.iter().all(|&b| b));
}

#[test]
fn randint_covers_its_half_open_interval() {
    let xs = sample(20_000, |k| S::randint(k, SCALAR, -5, 5));
    assert!(xs.iter().all(|&x| (-5..5).contains(&x)));
    assert!(xs.contains(&-5), "low bound must be reachable");
    assert!(xs.contains(&4), "high-1 must be reachable");
    assert!(!xs.contains(&5), "high bound is exclusive");
}

#[test]
fn randint_on_an_empty_interval_returns_the_bound() {
    let k = S::key(3);
    assert_eq!(S::randint(&k, SCALAR, 7, 7), 7);
}

#[test]
fn standard_gamma_has_mean_and_variance_equal_to_concentration() {
    // Gamma(a, 1) has mean a and variance a.
    for a in [0.5_f64, 1.0, 3.0, 7.5] {
        let xs = sample(20_000, |k| S::standard_gamma(k, SCALAR, a));
        assert!(xs.iter().all(|&x| x > 0.0), "support is positive");
        assert!(
            (mean(&xs) - a).abs() < 0.15 * a.max(1.0),
            "a={a}: mean was {}",
            mean(&xs)
        );
        assert!(
            (variance(&xs) - a).abs() < 0.25 * a.max(1.0),
            "a={a}: var was {}",
            variance(&xs)
        );
    }
}

#[test]
fn poisson_has_mean_and_variance_equal_to_its_rate() {
    for rate in [0.5_f64, 4.0, 12.0] {
        let xs: Vec<f64> = sample(20_000, |k| S::poisson(k, SCALAR, rate) as f64);
        assert!(xs.iter().all(|&x| x >= 0.0));
        assert!(
            (mean(&xs) - rate).abs() < 0.15 * rate.max(1.0),
            "rate={rate}: mean was {}",
            mean(&xs)
        );
        assert!(
            (variance(&xs) - rate).abs() < 0.3 * rate.max(1.0),
            "rate={rate}: var was {}",
            variance(&xs)
        );
    }
}

#[test]
fn binomial_has_mean_np_and_stays_in_range() {
    let (n, p) = (30.0_f64, 0.4);
    let xs: Vec<f64> = sample(20_000, |k| S::binomial(k, SCALAR, n, p) as f64);
    assert!(xs.iter().all(|&x| (0.0..=n).contains(&x)));
    assert!((mean(&xs) - n * p).abs() < 0.5, "mean was {}", mean(&xs));
    // Var = n p (1-p) = 7.2
    assert!((variance(&xs) - n * p * (1.0 - p)).abs() < 1.0);
}

#[test]
fn beta_binomial_is_overdispersed_relative_to_binomial() {
    // p ~ Beta(1,1) is uniform, so the count spreads much wider than Binomial(n, 0.5).
    let n = 30.0_f64;
    let bb: Vec<f64> = sample(20_000, |k| S::beta_binomial(k, SCALAR, n, 1.0, 1.0) as f64);
    let bin: Vec<f64> = sample(20_000, |k| S::binomial(k, SCALAR, n, 0.5) as f64);

    assert!(bb.iter().all(|&x| (0.0..=n).contains(&x)));
    assert!((mean(&bb) - n * 0.5).abs() < 1.0, "mean was {}", mean(&bb));
    assert!(
        variance(&bb) > 2.0 * variance(&bin),
        "beta-binomial variance {} should exceed binomial {}",
        variance(&bb),
        variance(&bin)
    );
}

#[test]
fn split_keys_decorrelate_draws() {
    // Draws off split children should look like an independent sample, not a repeated value.
    let children = S::split(&S::key(2024), 2_000);
    let xs: Vec<f64> = children
        .iter()
        .map(|k| S::uniform(k, SCALAR, 0.0, 1.0))
        .collect();
    assert!((mean(&xs) - 0.5).abs() < 0.03, "mean was {}", mean(&xs));
    assert!((variance(&xs) - 1.0 / 12.0).abs() < 0.01);
}
