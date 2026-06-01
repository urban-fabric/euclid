#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::{Count, ThreadCount};

#[test]
fn count_display_is_bare_number() {
    assert_eq!(Count::new(42).to_string(), "42");
}

#[test]
fn count_serde_is_bare_u64() {
    let c = Count::new(42);
    let json = serde_json::to_string(&c).unwrap();
    assert_eq!(json, "42");
    let back: Count = serde_json::from_str(&json).unwrap();
    assert_eq!(back, c);
}

#[test]
fn count_arithmetic_round_trips() {
    let mut c = Count::ZERO;
    c += Count::new(5);
    c += Count::new(3);
    assert_eq!(c, Count::new(8));
    c -= Count::new(1);
    assert_eq!(c, Count::new(7));
}

#[test]
fn thread_count_display_pluralizes() {
    assert_eq!(ThreadCount::new(1).to_string(), "1 thread");
    assert_eq!(ThreadCount::new(8).to_string(), "8 threads");
}

#[test]
fn thread_count_serde_is_bare_u32() {
    let t = ThreadCount::new(8);
    let json = serde_json::to_string(&t).unwrap();
    assert_eq!(json, "8");
    let back: ThreadCount = serde_json::from_str(&json).unwrap();
    assert_eq!(back, t);
}
