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
