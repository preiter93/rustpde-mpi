//! Steady state continuation
//!
//! export OPENBLAS_NUM_THREADS=1
//!
//! cargo run --release --example navier_rbc_steady_continuation
use rustpde::integrate;
use rustpde::navier_stokes::Navier2DAdjoint;
// use rustpde::Integrate;

fn main() {
    // Parameters
    let (nx, ny) = (128, 65);
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.5;
    let ra_list = ndarray::Array::logspace(10., 4.0, 4.2, 1);

    get_first_field();
    let mut restart = String::from("restart.h5");
    for ra in ra_list.iter() {
        // Setup
        let hdffile = format!("flow_ra{:4.2e}.h5", ra);
        let txtfile = format!("flow_ra{:4.2e}.txt", ra);

        if std::path::Path::new(&hdffile).exists() {
            println!("Skip Ra: {:?}", ra);
            restart = String::from(&hdffile);
            continue;
        }

        let mut navier = Navier2DAdjoint::new_periodic(nx, ny, *ra, pr, dt, aspect);
        navier.read(&restart);
        navier.reset_time();

        restart = String::from(&hdffile);

        // Solve
        integrate(&mut navier, 6000., Some(100.0));

        // Write
        navier.write(&hdffile);
        let nu = &navier.eval_nu();
        let nuvol = navier.eval_nuvol();
        let re = navier.eval_re();
        let data = format!("{:8.6e}  {:8.6e}  {:8.6e}  {:8.6e} \n", *ra, nu, nuvol, re);
        std::fs::write(txtfile, data).expect("Unable to write file");
    }
}

#[allow(dead_code)]
fn get_first_field() {
    use rustpde::mpi::initialize;
    use rustpde::mpi::integrate as int_mpi;
    use rustpde::navier_stokes_mpi::Navier2DMpi;
    // mpi
    let universe = initialize().unwrap();
    // Parameters
    let (nx, ny) = (128, 65);
    let ra = 1e4;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.1;
    let mut navier = Navier2DMpi::new_periodic(&universe, nx, ny, ra, pr, dt, aspect);
    navier.write_intervall = Some(100.0);
    navier.random_disturbance(1e-4);
    int_mpi(&mut navier, 100., Some(1.0));
    navier.write("restart.h5");
}
