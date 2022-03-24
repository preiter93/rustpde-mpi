//! Types and traits for real and complex numbers
pub use funspace::FloatNum;
pub use funspace::ScalarNum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Scalar type, used throughout this crate for arithmetic operations
pub trait Scalar: ScalarNum + SubAssign + AddAssign + MulAssign + DivAssign {}

impl<T> Scalar for T where T: ScalarNum + SubAssign + AddAssign + MulAssign + DivAssign {}
