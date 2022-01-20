//! Simulate Rayleigh-Benard Convection two dimensional
//! in a periodic domain
//!
//! cargo run --release --example navier_rbc_periodic
use rustpde::integrate;
use rustpde::navier_stokes::Navier2D;
// use rustpde::Integrate;

fn main() {
    // Parameters
    let (nx, ny) = (128, 129);
    let ra = 1e5;
    let pr = 1.;
    let aspect = 1.0; // corresponds to a  lateral length of 2pi
    let dt = 0.01;
    let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect, "rbc");
    //navier.read("restart.h5");
    //navier.reset_time();
    // Set initial conditions
    navier.set_velocity(0.2, 1., 1.);
    navier.set_temperature(0.2, 1., 1.);
    integrate(&mut navier, 10., Some(5.0));
}
