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
    let (nx, ny) = (1024, 256);
    let ra = 1e8;
    let pr = 10.;
    let aspect = 2.0;
    let dt = 0.001;
    let mut navier = Navier2DMpi::new_periodic_hc(&universe, nx, ny, ra, pr, dt, aspect);
    navier.read("data/flow00049.50.h5");
    // navier.reset_time();
    // Set initial conditions
    // navier.set_velocity(0.2, 1., 1.);
    // navier.set_temperature(0.2, 1., 1.);
    // navier.random_disturbance(1e-2);
    integrate(&mut navier, 400., Some(0.5));
}
