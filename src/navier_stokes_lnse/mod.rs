//! Linearized Navier-Stokes equations
#![allow(clippy::similar_names)]
pub mod functions;
pub mod lnse;
pub mod lnse_adj_eq;
pub mod lnse_adj_grad;
pub mod lnse_eq;
pub mod lnse_fd_grad;
pub mod lnse_io;
pub mod meanfield;
pub use lnse::Navier2DLnse;
pub use nonlin::Navier2DNonLin;
pub mod nonlin;
pub mod nonlin_adj_eq;
pub mod nonlin_adj_grad;
pub mod nonlin_eq;
pub mod nonlin_io;
pub mod opt_routines;
