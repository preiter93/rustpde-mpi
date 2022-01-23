//! Gradient based optimization routines
use super::functions::l2_norm;
use ndarray::Array2;

/// Steepest descent optimization without energy increase of the
/// target flow (energy constrainted)
///
/// # Panics
/// Input `alpha` > 2.
#[allow(
    clippy::shadow_unrelated,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]
pub fn steepest_descent_energy_constrained(
    velx_0: &Array2<f64>,
    vely_0: &Array2<f64>,
    temp_0: &Array2<f64>,
    grad_velx: &mut Array2<f64>,
    grad_vely: &mut Array2<f64>,
    grad_temp: &mut Array2<f64>,
    velx_new: &mut Array2<f64>,
    vely_new: &mut Array2<f64>,
    temp_new: &mut Array2<f64>,
    beta1: f64,
    beta2: f64,
    alpha: f64,
) {
    if alpha > 2. * std::f64::consts::PI {
        panic!("alpha must be less than 2 pi")
    }
    let n = velx_0.len() as f64;
    let e0 = l2_norm(velx_0, velx_0, vely_0, vely_0, temp_0, temp_0, beta1, beta2) / n;
    let eg = l2_norm(
        grad_velx, velx_0, grad_vely, vely_0, grad_temp, temp_0, beta1, beta2,
    ) / n;

    // Project gradient perpendicular to x0
    let ee = eg / e0;
    *grad_velx -= &(ee * velx_0);
    *grad_vely -= &(ee * vely_0);
    *grad_temp -= &(ee * temp_0);

    // Linear combination of old field and gradient
    let eg = l2_norm(
        grad_velx, grad_velx, grad_vely, grad_vely, grad_temp, grad_temp, beta1, beta2,
    ) / n;
    let ee2 = (e0 / eg).sqrt();
    velx_new
        .assign(&(velx_0.mapv(|x| x * alpha.cos()) + grad_velx.mapv(|x| x * ee2 * alpha.sin())));
    vely_new
        .assign(&(vely_0.mapv(|x| x * alpha.cos()) + grad_vely.mapv(|x| x * ee2 * alpha.sin())));
    temp_new
        .assign(&(temp_0.mapv(|x| x * alpha.cos()) + grad_temp.mapv(|x| x * ee2 * alpha.sin())));
}
