//! Steady state continuation
//!
//! export OPENBLAS_NUM_THREADS=1
//!
//! cargo run --release --example navier_rbc_steady

fn main() {
    use rustpde::integrate;
    use rustpde::navier_stokes::steady_adjoint::Navier2DAdjoint;

    // Parameters
    let (nx, ny) = (57, 57);
    let ra = 1e4;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2DAdjoint::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
    // PROVIDE INITIAL FLOW FIELD!
    navier.read_unwrap("restart.h5");
    navier.reset_time();
    integrate(&mut navier, 100., Some(1.0));
}

// fn generate_restart_field() {
//     use rustpde::integrate;
//     use rustpde::navier_stokes::Navier2D;
//     // Parameters
//     let (nx, ny) = (57, 57);
//     let ra = 1e4;
//     let pr = 1.;
//     let aspect = 1.0;
//     let dt = 0.1;
//     let mut navier = Navier2D::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
//     //navier.read_unwrap("restart.h5");
//     //navier.reset_time();
//     // navier.write_unwrap("restart.h5");
//     navier.init_random(1e-2);
//     integrate(&mut navier, 50., Some(2.0));
//     navier.write_unwrap("restart.h5");
// }
