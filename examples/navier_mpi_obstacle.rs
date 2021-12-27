//! Simulate Rayleigh-Benard Convection two dimensional
//! in a bounded domain with obstacle
//!
//! cargo mpirun --np 2 --example navier_mpi_obstacle --release
//!
//! Important: Disable obenblas multithreading:
//! export OPENBLAS_NUM_THREADS=1
use rustpde::mpi::initialize;
use rustpde::mpi::integrate;
// use rustpde::navier_stokes::solid_masks::solid_cylinder_inner;
use rustpde::navier_stokes::solid_masks::solid_rectangle;
use rustpde::navier_stokes_mpi::Navier2DMpi;

fn main() {
    // mpi
    let universe = initialize().unwrap();
    // Parameters
    let (nx, ny) = (129, 129);
    let ra = 1e4;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 0.71;
    let dt = 0.001;
    let mut navier = Navier2DMpi::new(&universe, nx, ny, ra, pr, dt, aspect, adiabatic);
    // Add obstacle
    // let solid = solid_cylinder_inner(&navier.field.x[0], &navier.field.x[1], 0., 0., 0.2);
    let solid = solid_rectangle(&navier.field.x[0], &navier.field.x[1], 0., 0.33, 0.8, 0.33);
    navier.solid = Some(solid);
    // navier.add_solid(solid);
    // //navier.read("restart.h5");
    // //navier.reset_time();
    // Set initial conditions
    navier.random_disturbance(1e-4);
    integrate(&mut navier, 10., Some(1.0));
}
