//! Run example:
//!
//! cargo mpirun --np 2 --bin rustpde --release
//!
//! Important: Disable obenblas multithreading:
//! ```
//! export OPENBLAS_NUM_THREADS=1
//! ```

// fn main() {
//     use rustpde::mpi::initialize;
//     use rustpde::mpi::integrate;
//     use rustpde::navier_stokes_mpi::Navier2DMpi;
//     // mpi
//     let universe = initialize().unwrap();
//     // Parameters
//     let (nx, ny) = (1024, 1025);
//     let ra = 1e4;
//     let pr = 1.;
//     let aspect = 1.0;
//     let dt = 0.01;
//     let max_time = 1.;
//     let mut navier = Navier2DMpi::new_periodic(&universe, nx, ny, ra, pr, dt, aspect, "rbc");
//     navier.init_random(1e-2);
//     // navier.read_unwrap("restart.h5");
//     // navier.reset_time();
//     let now = std::time::Instant::now();
//     integrate(&mut navier, max_time, Some(max_time));
//     if navier.nrank() == 0 {
//         println!("Time elapsed : {:?}", now.elapsed());
//         let iteration = max_time / dt;
//         println!("Per iteration: {:?} (s)", now.elapsed().as_secs_f64() / iteration);
//     }
// }

// SERIAL
fn main() {
    use rustpde::integrate;
    use rustpde::navier_stokes::Navier2D;
    // Parameters
    let (nx, ny) = (129, 129);
    let (ra, pr, aspect) = (1e7, 1., 1.);
    let dt = 2e-3;
    let max_time = 100.;
    let mut navier = Navier2D::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
    navier.init_random(1e-2);
    // navier.read_unwrap("restart.h5");
    // navier.reset_time();
    let now = std::time::Instant::now();
    println!("Start");
    integrate(&mut navier, max_time, Some(0.2));
    println!("Time elapsed: {:?}", now.elapsed());
    let iteration = max_time / dt;
    println!(
        "Per iteration: {:?} (s)",
        now.elapsed().as_secs_f64() / iteration
    );
}
