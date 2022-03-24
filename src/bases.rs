//! # Bases
//! `Funspace` now as independent package
//!
//! Implemented:
//! - `Chebyshev` (Orthonormal), see [`chebyshev()`]
//! - `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
//! - `ChebNeumann` (Composite), see [`cheb_neumann()`]
//! - `ChebDirichletNeumann` (Composite), see [`cheb_dirichlet_neumann()`]
//! - `FourierC2c` (Orthonormal), see [`fourier_c2c()`]
//! - `FourierR2c` (Orthonormal), see [`fourier_r2c()`]
pub use funspace::cheb_dirichlet;
pub use funspace::cheb_dirichlet_neumann;
pub use funspace::cheb_neumann;
pub use funspace::chebyshev;
pub use funspace::fourier_c2c;
pub use funspace::fourier_r2c;
pub use funspace::{BaseC2c, BaseKind, BaseR2c, BaseR2r};
pub use funspace::{BaseSpace, Space1, Space2};
pub use funspace::{FunspaceElemental, FunspaceExtended, FunspaceSize};
