//! Run example:
//!
//! cargo run --release
//!
//! Important: Disable obenblas multithreading:
//! ```
//! export OPENBLAS_NUM_THREADS=1
//! ```

fn main() {
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
}
