//! Run example:
//!
//! cargo mpirun --np 2 --bin rustpde --release
//!
//! Important: Disable obenblas multithreading:
//! ```
//! export OPENBLAS_NUM_THREADS=1
//! ```

fn maina() {
    use rustpde::integrate;
    use rustpde::navier_stokes::Navier2D;
    // Parameters
    let (nx, ny) = (129, 129);
    let ra = 1e5;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2D::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
    navier.init_random(1e-2);
    //navier.read_unwrap("restart.h5");
    //navier.reset_time();
    integrate(&mut navier, 140., Some(2.0));
    navier.write_unwrap("restart.h5");
}

fn main() {
    use rustpde::integrate;
    use rustpde::navier_stokes::steady_adjoint::Navier2DAdjoint;
    // Parameters
    let (nx, ny) = (129, 129);
    let ra = 2e7;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.01;
    let mt = 100000. * dt;
    let mut navier = Navier2DAdjoint::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
    navier.read_unwrap("restart.h5");
    navier.reset_time();
    integrate(&mut navier, mt, Some(mt / 100.));
}

// fn main() {
//     use rustpde::mpi::initialize;
//     use rustpde::mpi::integrate;
//     use rustpde::navier_stokes_mpi::Navier2DMpi;
//     // mpi
//     let universe = initialize().unwrap();
//     // Parameters
//     let (nx, ny) = (57, 57);
//     let ra = 1e4;
//     let pr = 1.;
//     let aspect = 1.0;
//     let dt = 0.01;
//     let mut navier = Navier2DMpi::new_confined(&universe, nx, ny, ra, pr, dt, aspect, "rbc");
//     //navier.read_unwrap("restart.h5");
//     //navier.reset_time();
//     // navier.write_unwrap("restart.h5");
//     navier.init_random(1e-2);
//     // .. integration time, output time
//     integrate(&mut navier, 1., Some(1.0));
// }
