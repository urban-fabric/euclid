//! Bounded complete tapes for finite domains and edge exploration.

use super::{Gen, Rc};

const LIMIT: usize = 4096;
const MAX_WORDS: usize = 1 << 18;
type Tapes = Vec<Vec<u64>>;
pub(super) type Metadata = (Option<Rc<Tapes>>, Rc<Tapes>);

fn fits(left: &Tapes, right: &Tapes) -> bool {
    left.len().saturating_add(right.len()) <= LIMIT
        && left
            .iter()
            .chain(right)
            .map(Vec::len)
            .fold(0usize, usize::saturating_add)
            <= MAX_WORDS
}

// The first raw word mapping to a particular index under multiply-high reduction.
fn word(index: u128, width: u128) -> u64 {
    ((index << 64).div_ceil(width)) as u64
}

pub(super) fn range_tapes(width: u128) -> Metadata {
    let finite = (width <= LIMIT as u128)
        .then(|| Rc::new((0..width).map(|index| vec![word(index, width)]).collect()));
    let mut edges = vec![vec![0]];
    if width > 1 {
        edges.push(vec![u64::MAX]);
    }
    if width > 2 {
        edges.push(vec![1 << 63]);
    }
    (finite, Rc::new(edges))
}

pub(super) fn product(left: Option<&Tapes>, right: Option<&Tapes>) -> Option<Tapes> {
    let (left, right) = (left?, right?);
    if left.len().checked_mul(right.len())? > LIMIT {
        return None;
    }
    let words_a: usize = left.iter().map(Vec::len).sum();
    let words_b: usize = right.iter().map(Vec::len).sum();
    if words_a
        .saturating_mul(right.len())
        .saturating_add(words_b.saturating_mul(left.len()))
        > MAX_WORDS
    {
        return None;
    }
    let mut result = Vec::new();
    for a in left {
        for b in right {
            let mut tape = a.clone();
            tape.extend_from_slice(b);
            result.push(tape);
        }
    }
    Some(result)
}

pub(super) fn sequence<T: 'static>(element: &Gen<T>, min: usize, max: usize) -> Metadata {
    // Bound metadata construction separately from actual generator sizes.
    if max > 64 {
        return (None, Rc::new(Vec::new()));
    }
    let (element_finite, element_edges) = element.exploration();
    let width = (max - min) as u128 + 1;
    let mut all = Some(Vec::new());
    let mut edges = Vec::new();
    let mut repeated = Some(vec![vec![]]);
    for len in 0..=max {
        if len >= min {
            let prefix = vec![vec![word((len - min) as u128, width)]];
            let cases = product(Some(&prefix), repeated.as_ref());
            all = match (all, cases) {
                (Some(mut a), Some(b)) if fits(&a, &b) => {
                    a.extend(b);
                    Some(a)
                }
                _ => None,
            };
            if len == min || len == max {
                if len == 0 {
                    edges.push(prefix[0].clone());
                }
                if len != 0 {
                    for element_tape in element_edges.iter() {
                        let words = 1usize.saturating_add(len.saturating_mul(element_tape.len()));
                        let used: usize = edges.iter().map(Vec::len).sum();
                        if edges.len() >= LIMIT || used.saturating_add(words) > MAX_WORDS {
                            break;
                        }
                        let mut tape = prefix[0].clone();
                        for _ in 0..len {
                            tape.extend_from_slice(element_tape);
                        }
                        edges.push(tape);
                    }
                }
            }
        }
        repeated = product(repeated.as_ref(), element_finite.as_deref());
    }
    (all.map(Rc::new), Rc::new(edges))
}

pub(super) fn weighted_tapes<T: 'static>(choices: &[(u32, Gen<T>)]) -> Metadata {
    let total: u128 = choices.iter().map(|(weight, _)| u128::from(*weight)).sum();
    assert!(total > 0, "weights must sum to a positive value");
    let mut offset = 0u128;
    let mut finite = Some(Vec::new());
    let mut edges = Vec::new();
    for (weight, generator) in choices {
        if *weight == 0 {
            continue;
        }
        let (generator_finite, generator_edges) = generator.exploration();
        let prefix = vec![vec![word(offset, total)]];
        finite = match (finite, product(Some(&prefix), generator_finite.as_deref())) {
            (Some(mut a), Some(b)) if fits(&a, &b) => {
                a.extend(b);
                Some(a)
            }
            _ => None,
        };
        if let Some(cases) = product(Some(&prefix), Some(&generator_edges))
            && fits(&edges, &cases)
        {
            edges.extend(cases);
        }
        offset += u128::from(*weight);
    }
    (finite.map(Rc::new), Rc::new(edges))
}
