//! Value-level tests for the eight capability extensions, run against [`Dense`].
//!
//! These exist because the capability traits previously had no implementation at all: they
//! type-checked, but nothing demonstrated that the signatures can express the operations they
//! name. Every assertion here is a hand-computed expected value.

use super::dense::Dense;
use crate::{
    FftKind, Tensor, TensorContract, TensorConv, TensorFft, TensorIndexing, TensorReduce,
    TensorSort, TensorTransform, TensorWindow,
};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn all_close(got: &Dense, want: &[f64]) -> bool {
    got.values().len() == want.len() && got.values().iter().zip(want).all(|(&g, &w)| close(g, w))
}

// --- TensorReduce -------------------------------------------------------

#[test]
fn cumsum_accumulates_along_the_axis() {
    let x = Dense::vector(&[1.0, 2.0, 3.0, 4.0]);
    assert!(all_close(&x.cumsum(0), &[1.0, 3.0, 6.0, 10.0]));
}

#[test]
fn cumsum_runs_per_lane_on_a_matrix() {
    let x = Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    // Along axis 1, each row accumulates independently.
    assert!(all_close(
        &x.clone().cumsum(1),
        &[1.0, 3.0, 6.0, 10.0, 30.0, 60.0]
    ));
    // Along axis 0, each column does.
    assert!(all_close(&x.cumsum(0), &[1.0, 2.0, 3.0, 11.0, 22.0, 33.0]));
}

#[test]
fn var_and_std_are_the_population_moments() {
    // mean 4, deviations -3,-1,1,3 -> var = (9+1+1+9)/4 = 5
    let x = Dense::vector(&[1.0, 3.0, 5.0, 7.0]);
    assert!(close(x.clone().var(&[]).item(), 5.0));
    assert!(close(x.std(&[]).item(), 5.0_f64.sqrt()));
}

#[test]
fn var_over_one_axis_removes_only_that_axis() {
    let x = Dense::matrix(2, 2, &[1.0, 3.0, 10.0, 14.0]);
    let v = x.var(&[1]);
    assert_eq!(v.dims(), &[2]);
    assert!(all_close(&v, &[1.0, 4.0]));
}

#[test]
fn logsumexp_is_stable_against_large_inputs() {
    let x = Dense::vector(&[1000.0, 1000.0]);
    // Naive exp() would overflow to inf here; the max-shift keeps it finite.
    let got = x.logsumexp(&[]).item();
    assert!(got.is_finite());
    assert!(close(got, 1000.0 + 2.0_f64.ln()));
}

#[test]
fn logsumexp_matches_the_direct_formula_for_small_inputs() {
    let xs = [0.5, -1.0, 2.0, 0.0];
    let want = xs.iter().map(|&x: &f64| x.exp()).sum::<f64>().ln();
    assert!(close(Dense::vector(&xs).logsumexp(&[]).item(), want));
}

#[test]
fn softmax_normalizes_each_lane_to_one() {
    let x = Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    let s = x.softmax(1);
    let rows: Vec<f64> = s.values().chunks(3).map(|r| r.iter().sum()).collect();
    assert!(rows.iter().all(|&r| close(r, 1.0)));
    // A uniform lane is exactly uniform.
    assert!(s.values()[3..].iter().all(|&v| close(v, 1.0 / 3.0)));
}

#[test]
fn log_softmax_is_the_log_of_softmax() {
    let x = Dense::vector(&[0.3, -1.2, 2.5]);
    let direct = x.clone().softmax(0).map_values(f64::ln);
    let stable = x.log_softmax(0);
    assert!(all_close(&stable, direct.values()));
}

// --- TensorSort ---------------------------------------------------------

#[test]
fn sort_orders_both_directions() {
    let x = Dense::vector(&[3.0, 1.0, 2.0]);
    assert!(all_close(&x.clone().sort(0, false), &[1.0, 2.0, 3.0]));
    assert!(all_close(&x.sort(0, true), &[3.0, 2.0, 1.0]));
}

#[test]
fn argsort_returns_the_ordering_permutation() {
    let x = Dense::vector(&[3.0, 1.0, 2.0]);
    assert!(all_close(&x.argsort(0, false), &[1.0, 2.0, 0.0]));
}

#[test]
fn argsort_composes_with_take_to_reproduce_sort() {
    // The documented relationship between the index and value families.
    let x = Dense::vector(&[5.0, -2.0, 9.0, 0.0]);
    let order = x.argsort(0, false);
    assert_eq!(x.take(&order, 0), x.clone().sort(0, false));
}

#[test]
fn argmax_and_argmin_find_the_extremes() {
    let x = Dense::vector(&[2.0, 9.0, 4.0, -1.0]);
    assert!(close(x.argmax(0).item(), 1.0));
    assert!(close(x.argmin(0).item(), 3.0));
}

#[test]
fn argmax_breaks_ties_toward_the_first_index() {
    let x = Dense::vector(&[1.0, 5.0, 5.0]);
    assert!(close(x.argmax(0).item(), 1.0));
}

#[test]
fn argmax_runs_per_row_on_a_matrix() {
    let x = Dense::matrix(2, 3, &[1.0, 7.0, 2.0, 9.0, 0.0, 3.0]);
    let got = x.argmax(1);
    assert_eq!(got.dims(), &[2]);
    assert!(all_close(&got, &[1.0, 0.0]));
}

#[test]
fn top_k_returns_values_and_their_indices() {
    let x = Dense::vector(&[4.0, 1.0, 9.0, 7.0]);
    let (values, indices) = x.top_k(2, true);
    assert!(all_close(&values, &[9.0, 7.0]));
    assert!(all_close(&indices, &[2.0, 3.0]));

    let (smallest, idx) = x.top_k(2, false);
    assert!(all_close(&smallest, &[1.0, 4.0]));
    assert!(all_close(&idx, &[1.0, 0.0]));
}

#[test]
fn top_k_clamps_k_to_the_axis_length() {
    let x = Dense::vector(&[1.0, 2.0]);
    let (values, _) = x.top_k(5, true);
    assert_eq!(values.dims(), &[2]);
}

#[test]
fn top_k_works_per_row_on_the_last_axis() {
    let x = Dense::matrix(2, 3, &[1.0, 5.0, 3.0, 9.0, 2.0, 4.0]);
    let (values, indices) = x.top_k(2, true);
    assert_eq!(values.dims(), &[2, 2]);
    assert!(all_close(&values, &[5.0, 3.0, 9.0, 4.0]));
    assert!(all_close(&indices, &[1.0, 2.0, 0.0, 2.0]));
}

// --- TensorContract -----------------------------------------------------

#[test]
fn dot_general_reproduces_matmul() {
    let a = Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Dense::matrix(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let via_dot = a.dot_general(&b, &[1], &[0], &[], &[]);
    let via_matmul = a.clone().matmul(b);
    assert_eq!(via_dot.dims(), &[2, 2]);
    assert_eq!(via_dot, via_matmul);
    // [[58,64],[139,154]]
    assert!(all_close(&via_dot, &[58.0, 64.0, 139.0, 154.0]));
}

#[test]
fn dot_general_batches_over_the_named_axes() {
    // Two independent 2x2 matmuls stacked on a leading batch axis.
    let a = Dense::new(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0]);
    let b = Dense::new(&[2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0, 8.0]);
    let got = a.dot_general(&b, &[2], &[1], &[0], &[0]);
    assert_eq!(got.dims(), &[2, 2, 2]);
    // batch 0: A·I = A ; batch 1: I·B = B
    assert!(all_close(&got, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
}

#[test]
fn dot_general_contracts_vectors_to_a_scalar() {
    let a = Dense::vector(&[1.0, 2.0, 3.0]);
    let b = Dense::vector(&[4.0, 5.0, 6.0]);
    let got = a.dot_general(&b, &[0], &[0], &[], &[]);
    assert!(got.dims().is_empty());
    assert!(close(got.item(), 32.0));
}

#[test]
fn einsum1_takes_a_diagonal() {
    let m = Dense::matrix(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert!(all_close(&m.einsum1("ii->i"), &[1.0, 5.0, 9.0]));
}

#[test]
fn einsum1_transposes_and_sums() {
    let m = Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = m.einsum1("ij->ji");
    assert_eq!(t.dims(), &[3, 2]);
    assert!(all_close(&t, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    // Dropping a letter sums over it.
    assert!(all_close(&m.einsum1("ij->j"), &[5.0, 7.0, 9.0]));
    assert!(close(m.einsum1("ij->").item(), 21.0));
}

#[test]
fn einsum2_spells_matmul_and_inner_product() {
    let a = Dense::matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Dense::matrix(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    assert_eq!(a.einsum2(&b, "ij,jk->ik"), a.clone().matmul(b));

    let u = Dense::vector(&[1.0, 2.0, 3.0]);
    let v = Dense::vector(&[4.0, 5.0, 6.0]);
    assert!(close(u.einsum2(&v, "i,i->").item(), 32.0));

    // Outer product keeps both letters.
    let outer = u.einsum2(&v, "i,j->ij");
    assert_eq!(outer.dims(), &[3, 3]);
    assert!(all_close(
        &outer,
        &[4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]
    ));
}

// --- TensorIndexing -----------------------------------------------------

#[test]
fn iota_is_the_index_range() {
    assert!(all_close(&Dense::iota(4), &[0.0, 1.0, 2.0, 3.0]));
}

#[test]
fn one_hot_encodes_each_index_as_a_row() {
    let idx = Dense::vector(&[0.0, 2.0, 1.0]);
    let got = Dense::one_hot(&idx, 3);
    assert_eq!(got.dims(), &[3, 3]);
    assert!(all_close(
        &got,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    ));
}

#[test]
fn one_hot_leaves_out_of_range_indices_empty() {
    let got = Dense::one_hot(&Dense::vector(&[5.0]), 3);
    assert!(all_close(&got, &[0.0, 0.0, 0.0]));
}

#[test]
fn one_hot_rows_sum_to_one_and_recover_the_index() {
    let idx = Dense::vector(&[0.0, 2.0, 1.0]);
    let encoded = Dense::one_hot(&idx, 3);
    // Summing the one-hot columns weighted by iota recovers the original index.
    assert_eq!(encoded.argmax(1), idx);
}

#[test]
fn bincount_histograms_the_indices() {
    let idx = Dense::vector(&[0.0, 1.0, 1.0, 3.0, 1.0]);
    assert!(all_close(
        &Dense::bincount(&idx, 5),
        &[1.0, 3.0, 0.0, 1.0, 0.0]
    ));
}

#[test]
fn bincount_totals_the_input_length() {
    let idx = Dense::vector(&[0.0, 1.0, 2.0, 2.0]);
    let counts = Dense::bincount(&idx, 3);
    assert!(close(counts.sum(&[]).item(), 4.0));
}

// --- TensorTransform ----------------------------------------------------

#[test]
fn roll_shifts_circularly_in_both_directions() {
    let x = Dense::vector(&[1.0, 2.0, 3.0, 4.0]);
    assert!(all_close(&x.roll(1, 0), &[4.0, 1.0, 2.0, 3.0]));
    assert!(all_close(&x.roll(-1, 0), &[2.0, 3.0, 4.0, 1.0]));
}

#[test]
fn roll_by_the_full_length_is_the_identity() {
    let x = Dense::vector(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(x.roll(4, 0), x);
    assert_eq!(x.roll(-8, 0), x);
}

#[test]
fn tile_repeats_along_the_axis() {
    let x = Dense::vector(&[1.0, 2.0]);
    let got = x.tile(3, 0);
    assert_eq!(got.dims(), &[6]);
    assert!(all_close(&got, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]));
}

#[test]
fn clip_bounds_every_element() {
    let x = Dense::vector(&[-5.0, 0.5, 9.0]);
    let got = x.clip(Dense::point(0.0), Dense::point(1.0));
    assert!(all_close(&got, &[0.0, 0.5, 1.0]));
}

// --- TensorWindow -------------------------------------------------------

#[test]
fn reduce_window_sum_slides_a_window() {
    let x = Dense::vector(&[1.0, 2.0, 3.0, 4.0]);
    // Width 2, stride 1, no padding -> 3 windows.
    let got = x.reduce_window_sum(&[2], &[1], &[(0, 0)]);
    assert_eq!(got.dims(), &[3]);
    assert!(all_close(&got, &[3.0, 5.0, 7.0]));
}

#[test]
fn reduce_window_respects_stride_and_padding() {
    let x = Dense::vector(&[1.0, 2.0, 3.0, 4.0]);
    // Stride 2 halves the output.
    assert!(all_close(
        &x.reduce_window_sum(&[2], &[2], &[(0, 0)]),
        &[3.0, 7.0]
    ));
    // One cell of padding on each side widens it again.
    let padded = x.reduce_window_sum(&[2], &[1], &[(1, 1)]);
    assert_eq!(padded.dims(), &[5]);
    assert!(all_close(&padded, &[1.0, 3.0, 5.0, 7.0, 4.0]));
}

#[test]
fn reduce_window_max_and_mean() {
    let x = Dense::vector(&[1.0, 5.0, 2.0, 8.0]);
    assert!(all_close(
        &x.reduce_window_max(&[2], &[1], &[(0, 0)]),
        &[5.0, 5.0, 8.0]
    ));
    assert!(all_close(
        &x.reduce_window_mean(&[2], &[1], &[(0, 0)]),
        &[3.0, 3.5, 5.0]
    ));
}

#[test]
fn max_pool2d_halves_an_nchw_image() {
    // 1x1x4x4 ramp, 2x2 pool, stride 2.
    let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let x = Dense::new(&[1, 1, 4, 4], data);
    let got = x.max_pool2d_nchw((2, 2), (2, 2), [(0, 0), (0, 0)]);
    assert_eq!(got.dims(), &[1, 1, 2, 2]);
    // Each 2x2 block's max is its bottom-right corner.
    assert!(all_close(&got, &[5.0, 7.0, 13.0, 15.0]));
}

#[test]
fn avg_pool2d_averages_each_block() {
    let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let x = Dense::new(&[1, 1, 4, 4], data);
    let got = x.avg_pool2d_nchw((2, 2), (2, 2), [(0, 0), (0, 0)]);
    assert!(all_close(&got, &[2.5, 4.5, 10.5, 12.5]));
}

// --- TensorConv ---------------------------------------------------------

#[test]
fn conv2d_with_an_identity_kernel_is_the_identity() {
    let x = Dense::new(&[1, 1, 3, 3], (1..=9).map(|i| i as f64).collect());
    let kernel = Dense::new(&[1, 1, 1, 1], vec![1.0]);
    let got = x.conv2d_nchw(&kernel, (1, 1), [(0, 0), (0, 0)], (1, 1), (1, 1), 1);
    assert_eq!(got.dims(), &[1, 1, 3, 3]);
    assert!(all_close(
        &got,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ));
}

#[test]
fn conv2d_sums_a_box_kernel() {
    let x = Dense::new(&[1, 1, 3, 3], (1..=9).map(|i| i as f64).collect());
    let kernel = Dense::new(&[1, 1, 2, 2], vec![1.0; 4]);
    let got = x.conv2d_nchw(&kernel, (1, 1), [(0, 0), (0, 0)], (1, 1), (1, 1), 1);
    assert_eq!(got.dims(), &[1, 1, 2, 2]);
    // 2x2 sliding sums over [[1,2,3],[4,5,6],[7,8,9]]
    assert!(all_close(&got, &[12.0, 16.0, 24.0, 28.0]));
}

#[test]
fn conv2d_same_padding_preserves_the_spatial_shape() {
    let x = Dense::new(&[1, 1, 3, 3], (1..=9).map(|i| i as f64).collect());
    let kernel = Dense::new(&[1, 1, 3, 3], vec![1.0; 9]);
    let got = x.conv2d_nchw(&kernel, (1, 1), [(1, 1), (1, 1)], (1, 1), (1, 1), 1);
    assert_eq!(got.dims(), &[1, 1, 3, 3]);
    // Centre tap covers the whole 3x3, summing 1..9.
    assert!(close(got.values()[4], 45.0));
}

#[test]
fn conv2d_stride_subsamples_the_output() {
    let x = Dense::new(&[1, 1, 4, 4], (0..16).map(|i| i as f64).collect());
    let kernel = Dense::new(&[1, 1, 2, 2], vec![1.0; 4]);
    let got = x.conv2d_nchw(&kernel, (2, 2), [(0, 0), (0, 0)], (1, 1), (1, 1), 1);
    assert_eq!(got.dims(), &[1, 1, 2, 2]);
    assert!(all_close(&got, &[10.0, 18.0, 42.0, 50.0]));
}

#[test]
fn conv2d_kernel_dilation_widens_the_receptive_field() {
    let x = Dense::new(&[1, 1, 3, 3], (1..=9).map(|i| i as f64).collect());
    // A dilated 2x2 kernel of ones reaches the four corners of the 3x3 input.
    let kernel = Dense::new(&[1, 1, 2, 2], vec![1.0; 4]);
    let got = x.conv2d_nchw(&kernel, (1, 1), [(0, 0), (0, 0)], (1, 1), (2, 2), 1);
    assert_eq!(got.dims(), &[1, 1, 1, 1]);
    assert!(close(got.item(), 1.0 + 3.0 + 7.0 + 9.0));
}

#[test]
fn conv2d_feature_groups_keep_channels_separate() {
    // Two input channels, two groups, one output channel each: no cross-channel mixing.
    let x = Dense::new(&[1, 2, 1, 2], vec![1.0, 2.0, 10.0, 20.0]);
    let kernel = Dense::new(&[2, 1, 1, 1], vec![1.0, 100.0]);
    let got = x.conv2d_nchw(&kernel, (1, 1), [(0, 0), (0, 0)], (1, 1), (1, 1), 2);
    assert_eq!(got.dims(), &[1, 2, 1, 2]);
    assert!(all_close(&got, &[1.0, 2.0, 1000.0, 2000.0]));
}

#[test]
fn conv_rejects_an_unsupported_dimension_numbering() {
    // This reference only implements NCHW/OIHW; anything else must not silently compute.
    let x = Dense::new(&[1, 1, 2, 2], vec![1.0; 4]);
    let k = Dense::new(&[1, 1, 1, 1], vec![1.0]);
    let got = x.conv_general_dilated(
        &k,
        &[1, 1],
        &[(0, 0), (0, 0)],
        &[1, 1],
        &[1, 1],
        3, // batch axis is not 0
        1,
        &[2, 3],
        1,
        0,
        &[2, 3],
        1,
        1,
    );
    assert!(got.values().is_empty());
}

// --- TensorFft ----------------------------------------------------------

/// Interleave real values as `re, im` pairs with zero imaginary part.
fn as_complex(re: &[f64]) -> Dense {
    let mut data = Vec::with_capacity(re.len() * 2);
    for &x in re {
        data.push(x);
        data.push(0.0);
    }
    Dense::new(&[data.len()], data)
}

#[test]
fn fft_of_a_constant_signal_is_a_single_spike() {
    // DFT of [1,1,1,1] is [4,0,0,0].
    let x = as_complex(&[1.0, 1.0, 1.0, 1.0]);
    let spectrum = x.fft_forward(&[4]);
    assert_eq!(spectrum.dims(), &[8]);
    assert!(close(spectrum.values()[0], 4.0));
    assert!(spectrum.values()[2..].iter().all(|&v| close(v, 0.0)));
}

#[test]
fn fft_matches_a_hand_computed_spectrum() {
    // DFT of [1,0,0,0] is flat: every bin is 1.
    let spectrum = as_complex(&[1.0, 0.0, 0.0, 0.0]).fft_forward(&[4]);
    for k in 0..4 {
        assert!(close(spectrum.values()[2 * k], 1.0));
        assert!(close(spectrum.values()[2 * k + 1], 0.0));
    }
}

#[test]
fn ifft_inverts_fft() {
    let x = as_complex(&[1.0, 2.0, -3.0, 0.5, 7.0, -1.0, 0.0, 4.0]);
    let round_trip = x.fft_forward(&[8]).ifft(&[8]);
    assert!(all_close(&round_trip, x.values()));
}

#[test]
fn fft_is_linear() {
    let a = as_complex(&[1.0, 2.0, 3.0, 4.0]);
    let b = as_complex(&[0.5, -1.0, 2.0, 0.0]);
    let sum_then_fft = (a.clone() + b.clone()).fft_forward(&[4]);
    let fft_then_sum = a.fft_forward(&[4]) + b.fft_forward(&[4]);
    assert!(all_close(&sum_then_fft, fft_then_sum.values()));
}

#[test]
fn fft_rejects_a_multi_axis_request() {
    // Only single-axis transforms are implemented; two lengths must not fabricate a result.
    let x = as_complex(&[1.0, 2.0, 3.0, 4.0]);
    assert!(x.fft(FftKind::Forward, &[2, 2]).values().is_empty());
}
