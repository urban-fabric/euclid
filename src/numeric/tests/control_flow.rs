use super::*;

/// A write-once bisection over [`LazyReal`]: drive a `(lo, hi, trip)` bracket toward the
/// root of a monotone-increasing `f` for a fixed trip count — the shape a generic
/// inverse-CDF quantile takes. The carriers whose mask is a host `bool` (`f64`/`f32`) run a
/// real `while` loop through [`LazyCompare::while_loop`]; a tensor backend lowers the
/// identical code to graph `While`.
fn bisect_increasing<T: LazyReal>(f: impl Fn(T) -> T, target: T, lo: T, hi: T, iters: u32) -> T {
    let trips = T::scalar(f64::from(iters));
    let (lo, hi, _) = T::while_loop(
        (lo, hi, T::zero()),
        |state: &(T, T, T)| state.2.lt(&trips),
        |(lo, hi, trip)| {
            let mid = (lo.clone() + hi.clone()) * T::half();
            let root_is_right = f(mid.clone()).lt(&target);
            let lo = T::select(&root_is_right, mid.clone(), lo);
            let hi = T::select(&root_is_right, hi, mid);
            (lo, hi, trip + T::one())
        },
    );
    (lo + hi) * T::half()
}

#[test]
fn lazy_real_while_loop_drives_bisection() {
    // √2 as the root of x² = 2 on [0, 2]: 60 host-loop trips reach full f64 precision.
    let root = bisect_increasing(|x| x.square(), 2.0_f64, 0.0, 2.0, 60);
    assert!((root - core::f64::consts::SQRT_2).abs() < 1e-12);

    // The identical write-once routine runs on f32.
    let root32 = bisect_increasing(|x| x.square(), 2.0_f32, 0.0, 2.0, 30);
    assert!((root32 - core::f32::consts::SQRT_2).abs() < 1e-6);

    // A zero trip count is the identity: the loop body never runs, so the initial
    // bracket midpoint is returned unchanged.
    assert_eq!(bisect_increasing(|x| x.square(), 2.0_f64, 0.0, 2.0, 0), 1.0);
}
