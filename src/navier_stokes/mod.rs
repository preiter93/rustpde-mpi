//! Navier--Stokes solver (no mpi)
pub mod adjoint;
pub mod boundary_conditions;
pub mod conv_term;
pub mod functions;
pub mod navier;
pub mod solid_masks;
pub mod statistics;
pub mod vorticity;
pub use adjoint::Navier2DAdjoint;
pub use navier::Navier2D;
//pub use conv_term::conv_term;
// pub use solid_masks::solid_cylinder_inner;
// pub use vorticity::vorticity_from_file;
