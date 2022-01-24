//! Calculate gradients of finite energy with
//! respect to inital field using finite differences
//! and adjoint based method. Compare gradients, to
//! validate adjoint gradient.
//!
//! cargo run --release --example navier_lnse_test_gradient
fn main() {
    use rustpde::navier_stokes_lnse::Navier2DLnse;

    // Navier parameter
    let (nx, ny) = (18, 13);
    let ra = 3e3;
    let pr = 0.1;
    let aspect = 1.;
    let dt = 0.01;
    let mut navier = Navier2DLnse::new_periodic(nx, ny, ra, pr, dt, aspect, "rbc");
    navier.init_random(1e-3);
    navier.write_unwrap("base.h5");

    let (_, mut grad_adj) = navier.grad_adjoint(10., Some(10.0), 0.5, 0.5, None);
    grad_adj.0.v *= -1.;
    grad_adj.1.v *= -1.;
    grad_adj.2.v *= -1.;

    navier.read_unwrap("base.h5");
    let grad_fd = navier.grad_fd(10., Some(10.0), 0.5, 0.5);

    approx_eq_norm(&grad_adj.0.v, &grad_fd.0.v);
    approx_eq_norm(&grad_adj.1.v, &grad_fd.1.v);
    approx_eq_norm(&grad_adj.2.v, &grad_fd.2.v);
}

fn approx_eq_norm<S>(
    a: &ndarray::ArrayBase<S, ndarray::Ix2>,
    b: &ndarray::ArrayBase<S, ndarray::Ix2>,
) where
    S: ndarray::Data<Elem = f64>,
{
    use rustpde::navier_stokes::functions::norm_l2_f64;
    let eps = 0.3;
    let norm_a = norm_l2_f64(a);
    let norm = norm_l2_f64(&(a - b));
    if norm / norm_a > eps {
        panic!("Relative {:?}", norm);
    } else {
        println!(
            "Diff is small: |g_fd - g_adj|/|g_adj| ) {:?}",
            norm / norm_a
        );
    }
}
