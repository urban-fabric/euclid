/// Explicit random-key management.
///
/// Keys are immutable values. A draw is a pure function of its key, distribution parameters, and
/// result shape; callers split or fold keys before independent draws.
pub trait RandomKey {
    /// The backend's key value. For graph/JIT backends this may be a traced value; for eager
    /// backends it may be ordinary host PRNG state.
    type Key: Clone;

    /// Build a reproducible key from a host seed.
    fn key(seed: u64) -> Self::Key;

    /// Deterministically split `key` into `count` independent child keys.
    fn split(key: &Self::Key, count: usize) -> Vec<Self::Key>;

    /// Deterministically mix host data into a key.
    ///
    /// Use this for loop indices, chain ids, epoch numbers, replica ids, or any other structured
    /// source of independent randomness.
    fn fold_in(key: &Self::Key, data: u64) -> Self::Key;
}
