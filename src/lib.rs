//! # `rustpde`: Spectral method solver for Navier-Stokes equations
//!<img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">
//!
//! # Dependencies
//! - cargo >= v1.49
//! - `hdf5` (sudo apt-get install -y libhdf5-dev)
//! - `clang` (only for parallel simulations, see [MPI](#mpi).)
//!
//! This version of `rustpde` contains serial and
//! mpi-parallel examples of fluid simulations using the spectral method.
//!
//! # `MPI`
//!
//! The mpi crate relies on a installed version of libclang. Also
//! make sure to add the clang bin folder to the path variable, i.e.
//! for example
//!
//! - `export PATH="${INSTALL_DIR}/llvm-project/build/bin:$PATH"`
//!
//! The correct mpi installation can be tricky at times. If you want
//! to use this library without mpi, you can disable of the default `mpi` feature.
//! Note that, if default features are turned off, do not forget to
//! specify which openblas backend you want to use. For example:
//!
//! - `cargo build --release --no-default-features --features openblas-static`
//!
//! # `OpenBlas`
//!
//! By default `rustpde` uses ndarray's `openblas-static` backend,
//! which is costly for compilation. To use a systems `OpenBlas`
//! installation, disable default features, and use the `openblas-system`
//! feature. Make sure to not forget to explicity use the `mpi` feature
//! in this case, .i.e.
//! - `cargo build --release --no-default-features --features mpi`
//!
//! Make sure the `OpenBlas` library is linked correctly in the library path,
//! i.e.
//!
//! - `export LIBRARY_PATH="${INSTALL_DIR}/OpenBLAS/lib"`
//!
//! IT IS NOT `LD_LIBRARY_PATH`!
//!
//! Openblas multithreading conflicts with internal multithreading.
//! Turn it off for better performance:
//! - `export OPENBLAS_NUM_THREADS=1`
//!
//! # `Hdf5`
//! Install Hdf5 and link as follows:
//!
//! - `export HDF5_DIR="${INSTALL_DIR}/hdf5-xx/" `
//! - ` export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${HDF5_DIR}/lib"`
//!
//! # Details
//!
//! Currently `rustpde` implements transforms from physical to spectral space
//! for the following basis functions:
//! - `Chebyshev` (Orthonormal), see [`bases::chebyshev()`]
//! - `ChebDirichlet` (Composite), see [`bases::cheb_dirichlet()`]
//! - `ChebNeumann` (Composite), see [`bases::cheb_neumann()`]
//! - `ChebDirichletNeumann` (Composite), see [`bases::cheb_dirichlet_neumann()`]
//! - `FourierR2c` (Orthonormal), see [`bases::fourier_r2c()`]
//! - `FourierC2c` (Orthonormal), see [`bases::fourier_c2c()`]
//!
//! Composite basis combine several basis functions of its parent space to
//! satisfy the boundary conditions, i.e. Galerkin method.
//!
//! ## Implemented solver
//!
//! - `2-D Rayleigh Benard Convection: Direct numerical simulation`,
//! see [`navier_stokes::Navier2D`] or  [`navier_stokes_mpi::Navier2DMpi`]
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
//!     let (nx, ny) = (129, 129);
//!     let (ra, pr, aspect) = (1e7, 1., 1.);
//!     let dt = 2e-3;
//!     let mut navier = Navier2DMpi::new_confined(&universe, nx, ny, ra, pr, dt, aspect, "rbc");
//!     navier.write_intervall = Some(1.0);
//!     integrate(&mut navier, 10., Some(0.1));
//! }
//!
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
//!     let (ra, pr, aspect) = (1e6, 1., 1.);
//!     let dt = 0.01;
//!     let mut navier = Navier2DMpi::new_periodic(&universe, nx, ny, ra, pr, dt, aspect, "rbc");
//!     navier.write_intervall = Some(1.0);
//!     integrate(&mut navier, 10., Some(0.1));
//! }
//! ```
//!
//! ## Postprocess the output
//!
//! `rustpde` contains some python scripts for postprocessing.
//! If you have run the above example and specified
//! to save snapshots, you will see `hdf5` files in the `data` folder.
//!
//! Plot a single snapshot via
//!
//! `python3 plot/plot2d.py`
//!
//! ### Paraview
//!
//! The xmf files, corresponding to the h5 files can be created
//! by the script
//!
//! `./tools/create_xmf`.
//!
//! This script works only for fields from the `Navier2D`
//! solver with the attributes temp, ux, uy and pres.
//! The tools folder contains also the full crate `create_xmf`, which
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
#[cfg(feature = "mpi")]
pub mod field_mpi;
pub mod io;
#[cfg(feature = "mpi")]
pub mod mpi;
pub mod navier_stokes;
pub mod navier_stokes_lnse;
#[cfg(feature = "mpi")]
pub mod navier_stokes_mpi;
pub mod solver;
#[cfg(feature = "mpi")]
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
