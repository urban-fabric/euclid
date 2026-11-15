//! End-to-end: the `property!` macro and the `check!` family, used the way a consumer uses them.

use crate::{assume, check, check_eq, check_ne, generate, property};

property! {
    fn a_range_generator_stays_in_range(v in generate::int(-10, 10)) {
        check!((-10..=10).contains(&v), "{v} escaped the range");
    }

    fn reversing_a_list_twice_is_the_identity(xs in generate::vec(generate::int(0, 50), 0, 10)) {
        let mut round_trip = xs.clone();
        round_trip.reverse();
        round_trip.reverse();
        check_eq!(round_trip, xs);
    }

    fn sorting_is_idempotent(xs in generate::vec(generate::int(0, 20), 0, 12)) {
        let mut once = xs.clone();
        once.sort_unstable();
        let mut twice = once.clone();
        twice.sort_unstable();
        check_eq!(once, twice);
    }

    fn two_generators_are_independent(
        a in generate::int(0, 5),
        b in generate::int(100, 105),
    ) {
        check!((0..=5).contains(&a));
        check!((100..=105).contains(&b));
        check_ne!(a, b, "disjoint ranges must never collide");
    }

    fn three_generators_bind_in_order(
        a in generate::int(0, 1),
        b in generate::int(2, 3),
        c in generate::int(4, 5),
    ) {
        check!(a < b && b < c, "bindings arrived out of order: {a} {b} {c}");
    }

    /// `assume!` skips a case without failing it.
    fn assume_skips_rather_than_fails(v in generate::int(0, 100)) {
        assume!(v % 2 == 0);
        check_eq!(v % 2, 0, "assume let an odd value through");
    }

    /// The dependent case, in the form the fleet actually uses it: a size, then indices bounded
    /// by that size.
    fn dependent_bounds_hold(
        case in generate::index(1, 6)
            .and_then(|n| generate::vec(generate::index(0, n - 1), 0, 5).map(move |xs| (n, xs)))
    ) {
        let (n, xs) = case;
        for x in &xs {
            check!(*x < n, "index {x} escaped a bound of {n}");
        }
    }
}

property! {
    #![config(crate::Config::default().cases(32))]

    /// A block-level config applies to every property in it. Thirty-two cases is enough to show
    /// the config took effect without being worth measuring.
    fn a_block_config_applies(v in generate::int(0, 3)) {
        check!((0..=3).contains(&v));
    }
}
