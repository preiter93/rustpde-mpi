//! Run example:
//!
//! cargo mpirun --np 2 --bin rustpde --release
//!
//! Important: Disable obenblas multithreading:
//! ```
//! export OPENBLAS_NUM_THREADS=1
//! ```
use rustpde::mpi::initialize;
use rustpde::mpi::integrate;
use rustpde::navier_stokes_mpi::Navier2DMpi;

fn main() {
    // mpi
    let universe = initialize().unwrap();
    // Parameters
    let (nx, ny) = (57, 57);
    let ra = 1e4;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2DMpi::new_confined(&universe, nx, ny, ra, pr, dt, aspect, "rbc");
    navier.read_unwrap("restart.h5");
    navier.reset_time();
    // navier.init_random(1e-2);
    integrate(&mut navier, 1., Some(0.2));
}
