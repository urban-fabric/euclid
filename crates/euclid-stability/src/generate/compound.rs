//! Compositional, full-width and size-aware generators.

use super::{DrawError, Gen, Rc, index, pair, vec, weighted};

/// An unsigned integer in the inclusive range `low..=high`.
pub fn uint(low: u64, high: u64) -> Gen<u64> {
    assert!(low <= high, "empty unsigned range");
    Gen::new(move |source| Ok(source.in_unsigned_range(low, high)?))
        .with_metadata(move || super::range_tapes(u128::from(high) - u128::from(low) + 1))
}

/// A full-width `u64`, including both endpoints.
pub fn u64() -> Gen<u64> {
    uint(0, u64::MAX)
}

/// A full-width `u128`. Two tape words, most significant first.
pub fn u128() -> Gen<u128> {
    pair(u64(), u64()).map(|(high, low)| (u128::from(high) << 64) + u128::from(low))
}

/// A full-width `i128`, ordered monotonically by the unsigned tape representation.
pub fn i128() -> Gen<i128> {
    u128().map(|bits| (bits ^ (1 << 127)) as i128)
}

/// One of a nonempty list of values. Small lists support exhaustive exploration.
pub fn sampled_from<T: Clone + 'static>(values: Vec<T>) -> Gen<T> {
    assert!(!values.is_empty(), "sampled_from needs a value");
    let width = values.len() as u128;
    let values = Rc::new(values);
    index(0, values.len() - 1)
        .map(move |pick| values[pick].clone())
        .with_metadata(move || {
            let (finite, edges) = super::range_tapes(width);
            let examples = finite.clone().unwrap_or(edges);
            (finite, examples)
        })
}

/// A fixed-length array. Its elements remain independently shrinkable.
pub fn array<T: 'static, const N: usize>(element: Gen<T>) -> Gen<[T; N]> {
    let elements = vec(element, N, N);
    let metadata_elements = elements.clone();
    Gen::new(move |source| {
        elements
            .draw(source)?
            .try_into()
            .map_err(|_| DrawError::Overrun)
    })
    .with_metadata(move || metadata_elements.exploration())
}

/// Optional values, with `None` at the smallest tape choice.
pub fn option<T: 'static>(element: Gen<T>) -> Gen<Option<T>> {
    // Construct None without imposing a Clone bound on T.
    let none = Gen::new(|_| Ok(None)).with_metadata(|| {
        (
            Some(Rc::new(std::vec![std::vec![]])),
            Rc::new(std::vec![std::vec![]]),
        )
    });
    weighted(std::vec![(1, none), (3, element.map(Some))])
}

/// Either successful or error values, with both branches represented in exploration.
pub fn result<T: 'static, E: 'static>(ok: Gen<T>, error: Gen<E>) -> Gen<Result<T, E>> {
    weighted(std::vec![(1, ok.map(Ok)), (1, error.map(Err))])
}

/// Draw a structural size in `0..=max`, then build a generator using that size.
///
/// The size is recorded on the tape, so replay and shrinking re-derive dependent bounds.
/// It controls structure, such as list length or tree depth, independently of numeric ranges.
pub fn sized<T: 'static>(max: usize, build: impl Fn(usize) -> Gen<T> + 'static) -> Gen<T> {
    index(0, max).and_then(build)
}
