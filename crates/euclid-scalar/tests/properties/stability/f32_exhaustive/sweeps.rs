//! The dense threshold sweep and the coarse breadth sweep. Part of [`super`].

use super::*;

/// Every `f32` around each kernel's saturation thresholds agrees with the wide peer's
/// classification.
///
/// This is the test that found all four defects. The thresholds are located per kernel by
/// [`crossing`] and the sweep is dense — every consecutive bit pattern within [`WINDOW`] — so
/// nothing in those regions is sampled over. The breadth sweep that follows is the coarse half,
/// and exists to catch anything that is *not* at a threshold.
#[test]
fn no_f32_kernel_saturates_where_the_wide_peer_does_not() {
    let rows = rows();
    let mut failures = Vec::new();
    let mut checked = 0u64;

    for row in &rows {
        let mut windows = Vec::new();
        if let Some(bits) = crossing(row, |w| w.abs() > f64::from(f32::MAX)) {
            windows.push(bits);
        }
        if let Some(bits) = crossing(row, |w| w.abs() < f64::from(f32::from_bits(1))) {
            windows.push(bits);
        }
        let radius = window();
        for centre in windows {
            let lo = centre.saturating_sub(radius);
            let hi = centre.saturating_add(radius).min(0x7F7F_FFFF);
            for bits in lo..=hi {
                for signed in [bits, bits | 0x8000_0000] {
                    let x = f32::from_bits(signed);
                    if !x.is_finite() || !(row.domain)(x) {
                        continue;
                    }
                    checked += 1;
                    if let Some(message) = disagreement(row, x).filter(|_| failures.len() < 20) {
                        failures.push(message);
                    }
                }
            }
        }
    }

    assert_eq!(rows.len(), 28, "the sweep lost kernels");
    assert!(
        checked >= 200_000,
        "only {checked} arguments checked; the threshold windows have collapsed"
    );
    assert!(
        failures.is_empty(),
        "f32 kernels disagreed with their wide peer's classification near a threshold:\n{}",
        failures.join("\n")
    );
}

/// The coarse breadth sweep, for anything that is not at a threshold.
///
/// Strided rather than dense, because its job is different: the dense sweep above is where the
/// arithmetic turns over, and this is everywhere else. `EUCLID_F32_STRIDE=1` makes it exhaustive —
/// the full `2.1e9`, tens of minutes — which is the only way to turn this file's claim from
/// "sampled and found nothing" into "there is nothing". The default samples one `f32` in `16,384`.
#[test]
fn the_f32_breadth_sweep_agrees_with_the_wide_peer() {
    let rows = rows();
    let stride = breadth_stride();
    let mut failures = Vec::new();
    let mut checked = 0u64;

    for row in &rows {
        let mut bits = 0u32;
        loop {
            for signed in [bits, bits | 0x8000_0000] {
                let x = f32::from_bits(signed);
                if !x.is_finite() || !(row.domain)(x) {
                    continue;
                }
                checked += 1;
                if let Some(message) = disagreement(row, x).filter(|_| failures.len() < 20) {
                    failures.push(message);
                }
            }
            let (next, overflowed) = bits.overflowing_add(stride);
            if overflowed || next > 0x7F7F_FFFF {
                break;
            }
            bits = next;
        }
    }

    assert!(
        checked >= 100_000,
        "only {checked} arguments checked at stride {stride}"
    );
    assert!(
        failures.is_empty(),
        "f32 kernels disagreed with their wide peer at stride {stride}:\n{}",
        failures.join("\n")
    );
}
