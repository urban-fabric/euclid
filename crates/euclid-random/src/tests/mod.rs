//! Contract tests, built on a reference backend.
//!
//! The crate ships only traits, so these tests exist to answer the question a contracts crate
//! has to answer: *is this surface actually implementable?* [`toy_sampler`] is a complete
//! eager-scalar implementation of all four traits, and the suites exercise it.

mod key_contract;
mod sampler_contract;
mod toy_sampler;
