//! Run example:
//!
//! cargo mpirun --np 2 --bin rustpde --release
//!
//! Important: Disable obenblas multithreading:
//! ```
//! export OPENBLAS_NUM_THREADS=1
//! ```

fn main() {
    use rustpde::mpi::initialize;
    use rustpde::mpi::integrate;
    use rustpde::navier_stokes_mpi::Navier2DMpi;
    // mpi
    let universe = initialize().unwrap();
    // Parameters
    let (nx, ny) = (33, 33);
    let ra = 1e4;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2DMpi::new_confined(&universe, nx, ny, ra, pr, dt, aspect, "rbc");
    navier.init_random(1e-2);
    // navier.read_unwrap("restart.h5");
    // navier.reset_time();
    integrate(&mut navier, 5., Some(1.0));
}

// SERIAL
// fn main() {
//     use rustpde::integrate;
//     use rustpde::navier_stokes::Navier2D;
//     // Parameters
//     let (nx, ny) = (33, 33);
//     let ra = 1e4;
//     let pr = 1.;
//     let aspect = 1.0;
//     let dt = 0.01;
//     let mut navier = Navier2D::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
//     navier.init_random(1e-2);
//     // navier.read_unwrap("restart.h5");
//     // navier.reset_time();
//     integrate(&mut navier, 5., Some(1.0));
// }
