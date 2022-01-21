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
    let (nx, ny) = (128, 57);
    let ra = 1e4;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.1;
    let mut navier = Navier2DMpi::new_periodic(&universe, nx, ny, ra, pr, dt, aspect, "rbc");
    // if navier.nrank() == 0 {
    //     navier.write("restart.h5");
    // }

    navier.read_unwrap("restart.h5");
    navier.reset_time();
    // navier.temp.scatter_spectral();
    // navier.temp.scatter_physical();
    // navier.temp.backward_mpi();
    // //
    // // Set initial conditions
    // // navier.random_disturbance(1e-2);
    integrate(&mut navier, 10., Some(1.0));
    // let temp = &navier.temp;
    // let axis = 0;
    //
    // let avga = temp.average_axis(axis);
    //
    // let avgb = temp.average_axis_mpi(axis);
    //
    // let avga = navier.eval_re();
    //
    // let avgb = navier.eval_re_mpi();
    //
    // if navier.nrank() == 0 {
    //     println!("{:?}", avga);
    //     println!("");
    // }
    //
    // if navier.nrank() == 1 {
    //     println!("{:?}", avgb);
    //     println!("");
    // }
}

// fn main() {
//     use rustpde::integrate;
//     use rustpde::navier_stokes::Navier2D;
//     // Parameters
//     let (nx, ny) = (128, 57);
//     let ra = 1e4;
//     let pr = 1.;
//     let aspect = 1.0;
//     let dt = 0.1;
//     let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect, "rbc");
//     // if navier.nrank() == 0 {
//     //     navier.write("restart.h5");
//     // }
//     navier.read_unwrap("restart.h5");
//
//     // Set initial conditions
//     // navier.random_disturbance(1e-2);
//     integrate(&mut navier, 10., Some(1.0));
// }
