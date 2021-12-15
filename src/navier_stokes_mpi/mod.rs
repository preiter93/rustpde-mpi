//! Navier--Stokes solver (mpi)
pub mod adjoint;
pub mod boundary_conditions;
pub mod functions;
pub mod navier;
pub mod statistics;
pub use adjoint::Navier2DAdjointMpi;
pub use navier::Navier2DMpi;
