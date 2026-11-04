//! `Dense`'s [`TensorConv`](crate::TensorConv) implementation.

use super::super::dense::Dense;
use crate::TensorConv;
impl TensorConv for Dense {
    /// Implemented for the `NCHW` input / `OIHW` kernel layout that
    /// [`conv2d_nchw`](TensorConv::conv2d_nchw) delegates to — batch/feature/spatial axes
    /// `0`/`1`/`[2,3]` on the input and output/input-feature axes `0`/`1` on the kernel, with
    /// strides, padding, both dilations, and feature grouping all honored. Other dimension
    /// numberings return an empty tensor rather than silently computing something else.
    #[allow(clippy::too_many_arguments)]
    fn conv_general_dilated(
        &self,
        rhs: &Self,
        window_strides: &[usize],
        padding: &[(usize, usize)],
        lhs_dilation: &[usize],
        rhs_dilation: &[usize],
        input_batch_dimension: usize,
        input_feature_dimension: usize,
        input_spatial_dimensions: &[usize],
        kernel_input_feature_dimension: usize,
        kernel_output_feature_dimension: usize,
        kernel_spatial_dimensions: &[usize],
        feature_group_count: usize,
        batch_group_count: usize,
    ) -> Self {
        let nchw = input_batch_dimension == 0
            && input_feature_dimension == 1
            && input_spatial_dimensions == [2, 3]
            && kernel_output_feature_dimension == 0
            && kernel_input_feature_dimension == 1
            && kernel_spatial_dimensions == [2, 3]
            && batch_group_count == 1;
        let ([n, c_in, h, w], [o, i_k, kh, kw]) = (self.dims(), rhs.dims()) else {
            return Dense::new(&[0], Vec::new());
        };
        if !nchw || feature_group_count == 0 || c_in % feature_group_count != 0 {
            return Dense::new(&[0], Vec::new());
        }
        let (n, c_in, h, w) = (*n, *c_in, *h, *w);
        let (o, i_k, kh, kw) = (*o, *i_k, *kh, *kw);
        if i_k != c_in / feature_group_count || o % feature_group_count != 0 {
            return Dense::new(&[0], Vec::new());
        }

        let stride = |a: usize| window_strides.get(a).copied().unwrap_or(1).max(1);
        let pad = |a: usize| padding.get(a).copied().unwrap_or((0, 0));
        let ldil = |a: usize| lhs_dilation.get(a).copied().unwrap_or(1).max(1);
        let rdil = |a: usize| rhs_dilation.get(a).copied().unwrap_or(1).max(1);

        // Dilation inflates the logical extents before the window slides.
        let eff = |len: usize, d: usize| if len == 0 { 0 } else { (len - 1) * d + 1 };
        let h_in = eff(h, ldil(0)) + pad(0).0 + pad(0).1;
        let w_in = eff(w, ldil(1)) + pad(1).0 + pad(1).1;
        let kh_eff = eff(kh, rdil(0));
        let kw_eff = eff(kw, rdil(1));
        if h_in < kh_eff || w_in < kw_eff {
            return Dense::new(&[0], Vec::new());
        }
        let h_out = (h_in - kh_eff) / stride(0) + 1;
        let w_out = (w_in - kw_eff) / stride(1) + 1;

        let per_group_out = o / feature_group_count;
        let mut data = vec![0.0; n * o * h_out * w_out];

        for b in 0..n {
            for oc in 0..o {
                let group = oc / per_group_out;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0.0;
                        for ic in 0..i_k {
                            let c = group * i_k + ic;
                            for r in 0..kh {
                                for s in 0..kw {
                                    let ph = oh * stride(0) + r * rdil(0);
                                    let pw = ow * stride(1) + s * rdil(1);
                                    // Undo padding, then undo input dilation.
                                    if ph < pad(0).0 || pw < pad(1).0 {
                                        continue;
                                    }
                                    let dh = ph - pad(0).0;
                                    let dw = pw - pad(1).0;
                                    if dh % ldil(0) != 0 || dw % ldil(1) != 0 {
                                        continue;
                                    }
                                    let (ih, iw) = (dh / ldil(0), dw / ldil(1));
                                    if ih >= h || iw >= w {
                                        continue;
                                    }
                                    let x =
                                        self.values()[Dense::ravel(&[b, c, ih, iw], self.dims())];
                                    let k = rhs.values()[Dense::ravel(&[oc, ic, r, s], rhs.dims())];
                                    acc += x * k;
                                }
                            }
                        }
                        data[Dense::ravel(&[b, oc, oh, ow], &[n, o, h_out, w_out])] = acc;
                    }
                }
            }
        }

        Dense::new(&[n, o, h_out, w_out], data)
    }
}
