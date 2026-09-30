//! Re-export topic constants for ergonomic imports.
//!
//! Usage:
//! ```rust
//! use predator_messages::topics::*;
//! println!("Subscribing to {}", LAYER1_DETECTION);
//! ```

pub use crate::zenoh_topics::*;
