//! Steady state continuation
//!
//! export OPENBLAS_NUM_THREADS=1
//!
//! cargo run --release --example navier_rbc_steady
// use rustpde::integrate;
// use rustpde::navier_stokes::Navier2DAdjoint;

fn main() {
    // // Parameters
    // let (nx, ny) = (64, 65);
    // let ra = 1e4;
    // let pr = 1.;
    // let aspect = 1.0;
    // let dt = 0.1;
    //
    // let mut navier = Navier2DAdjoint::new_periodic(nx, ny, ra, pr, dt, aspect);
    // navier.read("restart.h5");
    // navier.reset_time();
    // // Solve
    // integrate(&mut navier, 100., Some(5.0));
    println!("Currently unimplemented...");
}
