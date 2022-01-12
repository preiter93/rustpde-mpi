//! Run example:
//!
//! cargo mpirun --np 2 --example navier_periodic_hc_mpi --release
//!
//! Important: Disable obenblas multithreading:
//! export OPENBLAS_NUM_THREADS=1
use rustpde::mpi::initialize;
use rustpde::mpi::integrate;
use rustpde::navier_stokes_mpi::Navier2DMpi;

fn main() {
    // mpi
    let universe = initialize().unwrap();
    // Parameters
    let (nx, ny) = (128, 129);
    let ra = 1e5;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2DMpi::new_periodic_hc(&universe, nx, ny, ra, pr, dt, aspect);
    // navier.read("restart.h5");
    // navier.reset_time();
    // Set initial conditions
    navier.set_velocity(0.2, 1., 1.);
    navier.set_temperature(0.2, 1., 1.);
    // navier.random_disturbance(1e-2);
    integrate(&mut navier, 10., Some(5.));
}
