//! Run example:
//!
//! cargo mpirun --np 2 --bin rustpde --release
//!
//! Important: Disable obenblas multithreading:
//! ```
//! export OPENBLAS_NUM_THREADS=1
//! ```

// use rustpde::Integrate;

fn main() {
    use rustpde::integrate;
    use rustpde::navier_stokes::Navier2D;
    // Parameters
    let (nx, ny) = (128, 129);
    let ra = 1e5;
    let pr = 1.;
    let aspect = 1.0; // corresponds to a  lateral length of 2pi
    let dt = 0.01;
    let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
    //navier.read("restart.h5");
    //navier.reset_time();
    // Set initial conditions
    navier.set_velocity(0.2, 1., 1.);
    navier.set_temperature(0.2, 1., 1.);
    integrate(&mut navier, 10., Some(1.0));
}

fn main2() {
    use rustpde::bases::cheb_dirichlet;
    use rustpde::bases::fourier_r2c;
    use rustpde::field::{Field2, Space2};
    use rustpde::solver::HholtzAdi;

    let (nx, ny) = (10, 6);
    // let field = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny)));
    let field = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
    let (mat_a, mat_b, precond) = field.ingredients_for_hholtz(0);
    println!("{:?}", mat_b);

    let solver = HholtzAdi::new(&field, [0.1, 0.1]);
    println!("{:?}", solver.solver[0]);
}
