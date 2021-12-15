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
    let (nx, ny) = (64, 65);
    let ra = 1e4;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.1;
    let mut navier = Navier2DMpi::new_periodic(&universe, nx, ny, ra, pr, dt, aspect);
    navier.write_intervall = Some(1.0);
    navier.random_disturbance(1e-4);
    // navier.read("restart.h5");
    // navier.reset_time();
    integrate(&mut navier, 10., Some(0.1));
}
