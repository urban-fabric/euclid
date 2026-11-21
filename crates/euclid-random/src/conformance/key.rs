//! The key contract: splitting, derivation, and the independence they must produce.
//!
//! A counter-based generator's whole guarantee is that a key's children are independent of it
//! and of each other, so these laws are about *distinctness* rather than about any distribution
//! — a carrier that returns the same stream from two split keys is broken in a way no
//! goodness-of-fit test would flag.

use super::*;

/// The derivation laws: reproducibility, and that `split`/`fold_in` produce independent children
/// deterministically.
///
/// Nine laws, all observed through draws:
///
/// 1. the same seed draws the same value, twice over — the headline promise
/// 2. drawing does not disturb the key (it is a value, not a stream)
/// 3. distinct seeds draw distinct values
/// 4. `split` is a pure function of its key
/// 5. `split(k, n)` returns exactly `n` children
/// 6. `split(k, 0)` is empty
/// 7. children are pairwise independent, and none reproduces its parent
/// 8. `fold_in` is deterministic and depends on its data
/// 9. `split` and `fold_in` are different derivations
///
/// Laws 3 and 7 are *inequality* claims, so a correct-but-unlucky backend could in principle
/// fail them — a collision in the draw space is not a contract violation. In practice a
/// mixing function that decorrelates at all will not collide on a handful of seeds, and a
/// failure here is far more likely to be a missing domain separator than bad luck. If you hit
/// one, change the seeds and see whether it persists.
pub fn key_laws<R>(seeds: &[u64], what: &str)
where
    R: RandomSampler,
    R::Real: PartialEq + core::fmt::Debug,
{
    let draw = |k: &R::Key| R::uniform(k, SCALAR, R::Real::zero(), R::Real::one());

    for &s in seeds {
        // 1. same seed, same draw
        let k = R::key(s);
        assert!(
            draw(&k) == draw(&R::key(s)),
            "{what}: key({s}) must draw the same value every time"
        );

        // 2. the key is a value, not a consumed stream
        let first = draw(&k);
        let second = draw(&k);
        assert!(
            first == second,
            "{what}: drawing from a key must not disturb it"
        );

        // 4-7. split
        let children = R::split(&k, 4);
        assert_eq!(children.len(), 4, "{what}: split(k, 4) must return 4 keys");
        assert!(
            R::split(&k, 0).is_empty(),
            "{what}: split(k, 0) must be empty"
        );
        let again = R::split(&k, 4);
        for (i, (a, b)) in children.iter().zip(again.iter()).enumerate() {
            assert!(
                draw(a) == draw(b),
                "{what}: split must be a pure function of its key (child {i} differs)"
            );
        }
        for (i, c) in children.iter().enumerate() {
            assert!(
                draw(c) != first,
                "{what}: child {i} must not reproduce its parent's draw"
            );
            for (j, d) in children.iter().enumerate().skip(i + 1) {
                assert!(
                    draw(c) != draw(d),
                    "{what}: children {i} and {j} must be independent"
                );
            }
        }

        // 8. fold_in
        let folded = R::fold_in(&k, 11);
        assert!(
            draw(&folded) == draw(&R::fold_in(&k, 11)),
            "{what}: fold_in must be deterministic"
        );
        assert!(
            draw(&folded) != draw(&R::fold_in(&k, 12)),
            "{what}: fold_in must depend on its data"
        );
        assert!(
            draw(&folded) != first,
            "{what}: fold_in must not reproduce its parent's draw"
        );

        // 9. the two derivations are distinct
        assert!(
            draw(&R::split(&k, 1)[0]) != draw(&R::fold_in(&k, 0)),
            "{what}: split and fold_in must be different derivations"
        );
    }

    // 3. distinct seeds draw distinctly
    for (i, &a) in seeds.iter().enumerate() {
        for &b in seeds.iter().skip(i + 1) {
            assert!(
                draw(&R::key(a)) != draw(&R::key(b)),
                "{what}: seeds {a} and {b} must draw different values"
            );
        }
    }
}

/// [`key_laws`] over a backend's own [`ConformanceSample`].
pub fn key_laws_for<R>(what: &str)
where
    R: RandomSampler + ConformanceSample,
    R::Real: PartialEq + core::fmt::Debug,
{
    key_laws::<R>(&R::seeds(), what);
}
