//! # `rustpde`: Spectral method solver for Navier-Stokes equations
//!<img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">
//!
//! # Dependencies
//! - cargo >= v1.49
//! - `hdf5` (sudo apt-get install -y libhdf5-dev)
//!
//! This is the mpi version of `rustpde`. The following additional
//! dependencies are required:
//!
//! - mpi installation
//! - libclang
//!
//! # Important
//!
//! Openblas multithreading conflicts with internal multithreading.
//! Turn it off for better performance:
//!
//! # Details
//!
//! This library is intended for simulation softwares which solve the
//! partial differential equations using spectral methods.
//!
//! Currently `rustpde` implements transforms from physical to spectral space
//! for the following basis functions:
//! - `Chebyshev` (Orthonormal), see [`chebyshev()`]
//! - `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
//! - `ChebNeumann` (Composite), see [`cheb_neumann()`]
//! - `FourierR2c` (Orthonormal), see [`fourier_r2c()`]
//!
//! Composite basis combine several basis functions of its parent space to
//! satisfy the needed boundary conditions, this is often called a Galerkin method.
//!
//! ## Implemented solver
//!
//! - `2-D Rayleigh Benard Convection: Direct numerical simulation`,
//! see [`navier::navier`]
//! - `2-D Rayleigh Benard Convection: Steady state solver`,
//! see [`navier::navier_adjoint`]
//!
//! # Example
//! Solve 2-D Rayleigh Benard Convection ( Run with `cargo mpirun --np 2 --bin rustpde` )
//! ```ignore
//! use rustpde::mpi::initialize;
//! use rustpde::mpi::integrate;
//! use rustpde::navier_stokes_mpi::Navier2DMpi;
//!
//! fn main() {
//!     // mpi
//!     let universe = initialize().unwrap();
//!     // Parameters
//!     let (nx, ny) = (65, 65);
//!     let ra = 1e4;
//!     let pr = 1.;
//!     let adiabatic = true;
//!     let aspect = 1.0;
//!     let dt = 0.01;
//!     let mut navier = Navier2DMpi::new(&universe, nx, ny, ra, pr, dt, aspect, adiabatic);
//!     navier.write_intervall = Some(1.0);
//!     navier.random_disturbance(1e-4);
//!     integrate(&mut navier, 10., Some(0.1));
//! }
//! ```
//! Solve 2-D Rayleigh Benard Convection with periodic sidewall
//! ```ignore
//! use rustpde::mpi::initialize;
//! use rustpde::mpi::integrate;
//! use rustpde::navier_stokes_mpi::Navier2DMpi;
//!
//! fn main() {
//!     // mpi
//!     let universe = initialize().unwrap();
//!     // Parameters
//!     let (nx, ny) = (128, 65);
//!     let ra = 1e4;
//!     let pr = 1.;
//!     let adiabatic = true;
//!     let aspect = 1.0;
//!     let dt = 0.01;
//!     let mut navier = Navier2DMpi::new_periodic(&universe, nx, ny, ra, pr, dt, aspect);
//!     navier.write_intervall = Some(1.0);
//!     navier.random_disturbance(1e-4);
//!     integrate(&mut navier, 10., Some(0.1));
//! }
//! ```
//!
//! ## Postprocess the output
//!
//! `rustpde` contains a `python` folder with some scripts.
//! If you have run the above example and specified
//! to save snapshots, you will see `hdf5` files in the `data` folder.
//!
//! Plot a single snapshot via
//!
//! `python3 python/plot2d.py`
//!
//! or create an animation
//!
//! `python3 python/anim2d.py`
//!
//! ### Paraview
//!
//! The xmf files, corresponding to the h5 files can be created
//! by the script
//!
//! `./bin/create_xmf`.
//!
//! This script works only for fields from the `Navier2D`
//! solver with the attributes temp, ux, uy and pres.
//! The bin folder contains also the full crate `create_xmf`, which
//! can be adapted for specific usecases.
//!
//! ## Documentation
//!
//! Download and run:
//!
//! `cargo doc --open`
#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#[macro_use]
extern crate enum_dispatch;
pub mod bases;
pub mod field;
pub mod field_mpi;
pub mod io;
pub mod mpi;
pub mod navier_stokes;
pub mod navier_stokes_mpi;
pub mod solver;
pub mod solver_mpi;
pub mod types;

/// Real type (not active)
//pub type Real = f64;

const MAX_TIMESTEP: usize = 10_000_000;

/// Integrate trait, step forward in time, and write results
pub trait Integrate {
    /// Update solution
    fn update(&mut self);
    /// Receive current time
    fn get_time(&self) -> f64;
    /// Get timestep
    fn get_dt(&self) -> f64;
    /// Callback function (can be used for i/o)
    fn callback(&mut self);
    /// Additional break criteria
    fn exit(&mut self) -> bool;
}

/// Integrade pde, that implements the Integrate trait.
///
/// Specify `save_intervall` to force writing an output.
///
/// Stop Criteria:
/// 1. Timestep limit
/// 2. Time limit
pub fn integrate<T: Integrate>(pde: &mut T, max_time: f64, save_intervall: Option<f64>) {
    let mut timestep: usize = 0;
    let eps_dt = pde.get_dt() * 1e-4;
    loop {
        // Update
        pde.update();
        timestep += 1;

        // Save
        if let Some(dt_save) = &save_intervall {
            if (pde.get_time() % dt_save) < pde.get_dt() / 2.
                || (pde.get_time() % dt_save) > dt_save - pde.get_dt() / 2.
            {
                //println!("Save at time: {:4.3}", pde.get_time());
                pde.callback();
            }
        }

        // Break
        if pde.get_time() + eps_dt >= max_time {
            println!("time limit reached: {:?}", pde.get_time());
            break;
        }
        if timestep >= MAX_TIMESTEP {
            println!("timestep limit reached: {:?}", timestep);
            break;
        }
        if pde.exit() {
            println!("break criteria triggered");
            break;
        }
    }
}
