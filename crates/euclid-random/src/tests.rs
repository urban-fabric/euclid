//! Contract tests, built on a reference backend.
//!
//! The crate ships only traits, so these tests exist to answer the question a contracts crate
//! has to answer: *is this surface actually implementable?* [`toy_sampler`] is a complete
//! eager-scalar implementation of all four traits, and the suites exercise it.
//!
//! | module | what it establishes |
//! |--------|---------------------|
//! | [`key_contract`] | keys derive deterministically, and the public [`conformance`](crate::conformance) laws hold |
//! | [`sampler_contract`] | every distribution is implementable and respects its parameters |
//!
//! The key tests come in two kinds, and the distinction is worth knowing. Several assert on
//! keys directly — `assert_eq!` on a key, keys in a `HashSet` — which works only because
//! `ToySampler::Key` happens to be `Eq + Hash`, a property [`RandomKey`](crate::RandomKey)
//! does **not** require. Those check this key type. The conformance laws beside them observe
//! keys only through what they *draw*, which is what a different backend can run.

mod key_contract;
mod sampler_contract;
mod toy_sampler;
