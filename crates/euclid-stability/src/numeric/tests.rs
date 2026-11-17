//! The instruments, checked against functions whose behaviour is known by construction.
//!
//! Each finds a planted defect and leaves a clean function alone. A detector that fires on
//! everything is as useless as one that never fires, so both directions are required.

use super::{
    LIKELY_THRESHOLDS, UNIT_ROUNDOFF, condition_at, forward_error_bound, hunt_seams, next_down,
    next_up, scan_for_seam, scan_monotone, scan_window, seam_ratio, two_stage_error_bound,
};

mod conditioning;
mod representation;
mod seams;
