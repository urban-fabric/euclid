#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;

#[test]
fn bytes_display_carries_unit() {
    assert_eq!(Bytes(1024).to_string(), "1024 B");
}

#[test]
fn bytes_serde_is_bare_u64() {
    let b = Bytes(4096);
    let json = serde_json::to_string(&b).unwrap();
    assert_eq!(json, "4096");
    let back: Bytes = serde_json::from_str(&json).unwrap();
    assert_eq!(back, b);
}

#[test]
fn bytes_add_sub_round_trip() {
    let a = Bytes(100);
    let b = Bytes(40);
    assert_eq!((a + b).as_u64(), 140);
    assert_eq!((a - b).as_u64(), 60);
    let mut c = a;
    c += b;
    assert_eq!(c.as_u64(), 140);
    c -= b;
    assert_eq!(c.as_u64(), 100);
}

#[test]
fn byte_range_end_and_emptiness() {
    let r = ByteRange::new(Bytes(16), Bytes(32));
    assert_eq!(r.end(), Bytes(48));
    assert!(!r.is_empty());
    assert!(ByteRange::empty(Bytes(7)).is_empty());
}

#[test]
fn byte_range_display_uses_half_open_form() {
    let r = ByteRange::new(Bytes(16), Bytes(32));
    assert_eq!(r.to_string(), "[16..48) (32 B)");
}

#[test]
fn byte_range_serde_round_trip() {
    let r = ByteRange::new(Bytes(16), Bytes(32));
    let json = serde_json::to_string(&r).unwrap();
    let back: ByteRange = serde_json::from_str(&json).unwrap();
    assert_eq!(back, r);
}
