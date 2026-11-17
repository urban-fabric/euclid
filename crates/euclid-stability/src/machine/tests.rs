//! The sequential machinery, and one real application of it.
//!
//! The application is [`persist`](crate::persist) driven as a state machine, which is not a toy
//! chosen to demonstrate the feature — it is the code in this crate with the most genuine
//! multi-entry state, and the code where a bug of exactly this shape was found by hand. `refresh`
//! originally rewrote a regressions file from one property's point of view and deleted its
//! siblings' entries; a regressions file holds a whole suite, so that silently destroyed the only
//! record of cases that had cost a search to find.
//!
//! That bug is a *sequence* bug. No single call to `save` or `refresh` is wrong in isolation; the
//! failure needs a save under one name, a save under another, and then a refresh. A single-shot
//! property cannot express it and the machine below finds it immediately.

use std::fs;

use super::{Machine, check_machine, commands, execute};
use crate::persist::{self, Entry};
use crate::runner::{Config, Failure, Outcome};
use crate::{Gen, generate};

mod persistence;
mod sequences;
