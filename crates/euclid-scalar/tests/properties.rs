//! Generated laws and numerical stability searches on the public scalar surface.
//!
//! Run the whole suite with `cargo test -p euclid-scalar --test properties`.
//! Persisted tapes and budgets stay in `tests/regressions` and `tests/budgets.txt`.

#[path = "properties/conditioning.rs"]
mod conditioning;
#[path = "properties/gamma.rs"]
mod gamma;
#[path = "properties/identities.rs"]
mod identities;
#[path = "properties/stability.rs"]
mod stability;
