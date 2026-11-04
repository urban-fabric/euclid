//! **Which convention does it follow?** Pins the semantics of the std-mirroring elementary
//! methods, where more than one defensible answer exists.
//!
//! These 24 methods (`sin`, `floor`, `round`, `min`, `signum`, `atan2`, …) share their names
//! with `f64`'s inherent methods and currently carry no doc comments on the trait, on the
//! reasonable assumption that "it does what `f64` does". That assumption is safe for a
//! consumer *calling* them and unsafe for a consumer *implementing* them: an expression or
//! graph carrier has to choose a rounding mode, a NaN policy, and a branch convention, and
//! nothing in the contract currently says which.
//!
//! An unstated convention is the one failure mode that defeats the whole library — two
//! backends resolving it differently means the same formula yields different answers on
//! different interfaces. So this suite fixes the intended behavior as executable
//! specification, and doubles as the list of what the trait docs should say.
//!
//! Split by the kind of convention under test.

mod branches;
mod composites;
mod f32_parity;
mod nan_policy;
mod rounding;
mod signs;
