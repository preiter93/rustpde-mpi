//! Steady state calculation
//!
//! cargo mpirun --np 2 --example navier_rbc_steady_mpi --release
//!
//! Important: Disable obenblas multithreading:
//! export OPENBLAS_NUM_THREADS=1
use rustpde::mpi::initialize;
use rustpde::mpi::integrate;
use rustpde::mpi::navier::adjoint::Navier2DAdjointMpi;

fn main() {
    // mpi
    let universe = initialize().unwrap();
    // Parameters
    let (nx, ny) = (64, 65);
    let ra = 1e4;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.1;

    let mut navier = Navier2DAdjointMpi::new_periodic(&universe, nx, ny, ra, pr, dt, aspect);
    // Start from some inintal field
    navier.read("restart.h5");
    navier.reset_time();
    // Solve
    integrate(&mut navier, 100., Some(5.0));
}
