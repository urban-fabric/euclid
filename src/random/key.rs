use super::RandomAlgorithm;

/// Explicit random-key management.
///
/// Keys are immutable values. Draw methods in the capability traits below do not advance or
/// consume the key; callers split keys before independent draws, or fold deterministic data such
/// as loop steps / chain ids into a key.
/// This mirrors JAX's PRNG discipline and is the shape compiled backends can represent directly.
pub trait KeyedRandomBackend {
    /// The backend's key value. For graph backends this is usually an integer tensor.
    type Key: Clone;
    /// Serializable/interoperable key payload.
    ///
    /// Typed-key implementations should include enough information here to reconstruct the same
    /// key and RNG algorithm. Legacy backends may use raw words such as `[u32; 2]`.
    type KeyData: Clone;
    /// The random-bit algorithm, when the implementation exposes one.
    fn algorithm() -> RandomAlgorithm {
        RandomAlgorithm::Default
    }

    /// Build a reproducible key from a host seed.
    fn key(seed: u64) -> Self::Key;

    /// Extract an interoperable key payload.
    fn key_data(key: &Self::Key) -> Self::KeyData;

    /// Rebuild a key from an interoperable payload.
    fn wrap_key_data(data: Self::KeyData) -> Self::Key;

    /// Deterministically split `key` into `count` independent child keys.
    fn split(key: &Self::Key, count: usize) -> Vec<Self::Key>;

    /// Deterministically split `key` into a row-major array of child keys.
    fn split_shape(key: &Self::Key, shape: &[usize]) -> Vec<Self::Key> {
        let count = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .expect("random split shape element count overflow");
        Self::split(key, count)
    }

    /// Deterministically mix host data into a key.
    ///
    /// Use this for loop indices, replica ids, chain ids, epoch numbers, or any other structured
    /// source of independent randomness.
    fn fold_in(key: &Self::Key, data: u64) -> Self::Key;
}

/// Graph-native explicit random-key management.
///
/// Backends with a tracing/JIT layer can represent keys as graph values, accept
/// them as parameters, split/fold them inside a compiled computation, and return
/// key data as graph outputs. Eager backends can ignore this trait.
pub trait GraphRandomBackend: KeyedRandomBackend {
    /// A key value living in the backend's graph/tracing substrate.
    type GraphKey: Clone;
    /// Graph value carrying key data, usually unsigned integer words.
    type GraphKeyData: Clone;
    /// Scalar graph integer accepted by `graph_fold_in_value`.
    type GraphFoldData: Clone;

    /// Lift a host key into the active graph.
    fn graph_key(key: &Self::Key) -> Self::GraphKey;

    /// Create a graph parameter/input representing a key.
    fn graph_key_parameter(index: usize) -> Self::GraphKey;

    /// Return graph key data suitable for returning from compiled computations.
    fn graph_key_data(key: &Self::GraphKey) -> Self::GraphKeyData;

    /// Rebuild a graph key from graph key data.
    fn graph_wrap_key_data(data: Self::GraphKeyData) -> Self::GraphKey;

    /// Split a graph key into `count` scalar graph keys.
    fn graph_split(key: &Self::GraphKey, count: usize) -> Vec<Self::GraphKey>;

    /// Split a graph key into a row-major array of graph keys.
    fn graph_split_shape(key: &Self::GraphKey, shape: &[usize]) -> Vec<Self::GraphKey> {
        let count = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .expect("graph random split shape element count overflow");
        Self::graph_split(key, count)
    }

    /// Mix host data into a graph key.
    fn graph_fold_in(key: &Self::GraphKey, data: u64) -> Self::GraphKey;

    /// Mix scalar graph data into a graph key.
    fn graph_fold_in_value(key: &Self::GraphKey, data: Self::GraphFoldData) -> Self::GraphKey;
}
