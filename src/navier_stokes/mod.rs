//! Navier--Stokes solver (no mpi)
// pub mod adjoint;
pub mod boundary_conditions;
pub mod functions;
pub mod navier;
pub mod solid_masks;
pub mod statistics;
pub mod vorticity;
pub use navier::Navier2D;
pub mod navier_eq;
pub mod navier_io;
// pub use adjoint::Navier2DAdjoint;
