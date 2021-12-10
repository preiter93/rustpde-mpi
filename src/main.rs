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
// use rustpde::mpi::navier::statistics::Statistics;
use rustpde::mpi::navier::Navier2DMpi;

fn main() {
    // mpi
    let universe = initialize().unwrap();
    // Parameters
    let (nx, ny) = (64, 65);
    let ra = 1e4;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2DMpi::new_periodic(&universe, nx, ny, ra, pr, dt, aspect);
    navier.write_intervall = Some(1.0);
    // Statistics::new(navier, save_state, write_stat)
    // navier.statistics = Some(Statistics::new(&navier, 0.5, 10.0));
    // navier.read("data/flow00026.40.h5");
    // navier.reset_time();
    // Set initial conditions
    // navier.set_velocity(0.2, 1., 1.);
    // navier.set_temperature(0.2, 1., 1.);
    navier.random_disturbance(1e-4);
    integrate(&mut navier, 100., Some(0.1));
}
