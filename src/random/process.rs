use super::KeyedRandomBackend;

/// A distribution sampled from an explicit-key backend.
pub trait SampleWithKey<B: KeyedRandomBackend + ?Sized> {
    type Output;

    fn sample_with_key(&self, key: &B::Key) -> Self::Output;
}

/// A random process with explicit non-random state and explicit keys.
pub trait KeyedRandomProcess<B: KeyedRandomBackend + ?Sized> {
    type State;
    type Output;

    fn sample_step(&self, state: Self::State, key: &B::Key) -> (Self::State, Self::Output);
}
