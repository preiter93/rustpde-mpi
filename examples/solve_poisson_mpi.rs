//! Run example:
//!
//! cargo mpirun --np 2 --example solve_poisson_mpi --release
//!
//! Important: Disable obenblas multithreading:
//! export OPENBLAS_NUM_THREADS=1
use rustpde::bases::cheb_dirichlet;
use rustpde::field_mpi::{Field2Mpi, Space2Mpi};
use rustpde::mpi::initialize;
use rustpde::solver::Solve;
use rustpde::solver_mpi::poisson::PoissonMpi;

fn main() {
    // parameter
    let (nx, ny) = (257, 257);
    // mpi
    let universe = initialize().unwrap();
    // Setup space and field
    let space = Space2Mpi::new(
        &cheb_dirichlet::<f64>(nx),
        &cheb_dirichlet::<f64>(ny),
        &universe,
    );
    let mut field = Field2Mpi::new(&space);
    let x = &field.x[0];
    let y = &field.x[1];

    // Setup hholtz solver
    let alpha = 1.0;
    let poisson = PoissonMpi::new(&field, [alpha, alpha]);

    // Setup rhs and solution
    let n = std::f64::consts::PI / 2.;
    let mut expected = field.v.clone();
    for (i, xi) in x.iter().enumerate() {
        for (j, yi) in y.iter().enumerate() {
            field.v[[i, j]] = (n * xi).cos() * (n * yi).cos();
            expected[[i, j]] = -1. / (n * n * 2.) * field.v[[i, j]];
        }
    }

    // Solve
    field.forward();
    field.scatter_spectral();
    let rhs = field.to_ortho_mpi();
    for _ in 0..2000 {
        poisson.solve(&rhs, &mut field.vhat_x_pen, 0);
    }
    field.backward_mpi();
    field.all_gather_physical();

    // Compare
    approx_eq(&field.v, &expected);
}

fn approx_eq<S, D>(result: &ndarray::ArrayBase<S, D>, expected: &ndarray::ArrayBase<S, D>)
where
    S: ndarray::Data<Elem = f64>,
    D: ndarray::Dimension,
{
    let dif = 1e-3;
    for (a, b) in expected.iter().zip(result.iter()) {
        if (a - b).abs() > dif {
            panic!("Large difference of values, got {} expected {}.", b, a)
        }
    }
}
