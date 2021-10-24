//! Run example:
//!
//! cargo mpirun --np 2 --example solve_navier_mpi --release
//!
//! Important: Disable obenblas multithreading:
//! export OPENBLAS_NUM_THREADS=1
use rustpde::mpi::initialize;
use rustpde::mpi::integrate;
use rustpde::mpi::navier::Navier2DMpi;

fn main() {
    // mpi
    let universe = initialize().unwrap();
    // Parameters
    let (nx, ny) = (513, 513);
    let ra = 1e9;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.0002;
    let mut navier = Navier2DMpi::new(&universe, nx, ny, ra, pr, dt, aspect, adiabatic);
    navier.read("data/flow00026.40.h5");
    // navier.reset_time();
    // Set initial conditions
    // navier.set_velocity(0.2, 1., 1.);
    // navier.set_temperature(0.2, 1., 1.);
    //navier.random_disturbance(1e-4);
    integrate(&mut navier, 100., Some(0.1));
}
