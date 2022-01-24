//! Gradient based optimization to optimial disturbance
//! to trigger flow transitions
//!
//! cargo run --release --example navier_lnse_opt_reversals

use ndarray::{s, Array2};
fn mirror_field(velx: &Array2<f64>, vely: &Array2<f64>, temp: &Array2<f64>) -> [Array2<f64>; 3] {
    let velx_mirr = -1. * &velx.to_owned().slice(s![..;-1,..]);
    let vely_mirr = -1. * vely.to_owned();
    let temp_mirr = 1. * &temp.to_owned().slice(s![..;-1,..]);
    [velx_mirr, vely_mirr, temp_mirr]
}

fn find_base_field(nx: usize, ny: usize, dt: f64, ra: f64, pr: f64, aspect: f64) {
    use rustpde::integrate;
    use rustpde::navier_stokes::Navier2D;

    let mut navier = Navier2D::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
    navier.init_random(1e-3);
    integrate(&mut navier, 300., Some(300.0));
    navier.write_unwrap("mean.h5");
}

fn main() {
    use rustpde::navier_stokes_lnse::functions::l2_norm;
    use rustpde::navier_stokes_lnse::lnse_adj_grad::MAXIMIZE;
    use rustpde::navier_stokes_lnse::opt_routines::steepest_descent_energy_constrained;
    use rustpde::navier_stokes_lnse::Navier2DNonLin;

    // Opt parameter
    let alpha_0 = 1.0; // Step width gradient descent
    let (beta1, beta2) = (0.5, 0.5); // Energy weights (vel, temp)
    let max_iter = 30;

    let ti_list = ndarray::Array::linspace(5., 50., 5);
    let en_list = ndarray::Array::logspace(10., 0., 3., 7);

    // Navier parameter
    let (nx, ny) = (128, 57);
    let ra = 1e5;
    let pr = 1.;
    let aspect = 1.;
    let dt = 0.02;

    // Find initial large scale circulation
    find_base_field(nx, ny, dt, ra, pr, aspect);

    // Iterate over maximum time horizon and
    // initial energy of perturbances
    let mut funval = 0.;
    for &max_time in &ti_list {
        println!("MAX TIME: {:?}", max_time);
        for &energy_constraint in &en_list {
            println!("ENERGY: {:?}", energy_constraint);

            let flofile = format!(
                "flow_energy{:4.2e}_maxtime{:4.2e}.h5",
                energy_constraint, max_time
            );
            let txtfile = format!(
                "flow_energy{:4.2e}_maxtime{:4.2e}.txt",
                energy_constraint, max_time
            );

            if std::path::Path::new(&txtfile).exists() {
                println!("File {:?} exists. Skip.", txtfile);
                continue;
            }

            // Setup solver
            let mut navier = Navier2DNonLin::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
            navier.init_random(1e-3);
            // navier.write_unwrap("base.h5");
            // navier.read_unwrap("data/opt_field.h5");

            // Test with target state
            let mut target = navier.mean.clone();
            // mirror field, try to observe flow reversal
            let [velx_mirr, vely_mirr, temp_mirr] =
                mirror_field(&target.velx.v, &target.vely.v, &target.temp.v);
            target.velx.v = &velx_mirr - &navier.mean.velx.v;
            target.vely.v = &vely_mirr - &navier.mean.vely.v;
            target.temp.v = &temp_mirr - &navier.mean.temp.v;
            target.velx.forward();
            target.vely.forward();
            target.temp.forward();
            target.write_unwrap("data/target.h5");
            let target = Some(&target);

            // Scale energy
            let e0 = l2_norm(
                &navier.velx.v,
                &navier.velx.v,
                &navier.vely.v,
                &navier.vely.v,
                &navier.temp.v,
                &navier.temp.v,
                beta1,
                beta2,
            );
            navier.velx.v *= (energy_constraint / e0).sqrt();
            navier.vely.v *= (energy_constraint / e0).sqrt();
            navier.temp.v *= (energy_constraint / e0).sqrt();
            navier.velx.forward();
            navier.vely.forward();
            navier.temp.forward();

            // Normalize constant for later
            let norm = if let Some(t) = &target {
                let velx = &(&navier.velx.v - &t.velx.v);
                let vely = &(&navier.vely.v - &t.vely.v);
                let temp = &(&navier.temp.v - &t.temp.v);
                l2_norm(&velx, &velx, &vely, &vely, &temp, &temp, beta1, beta2)
            } else {
                let velx = &navier.velx.v;
                let vely = &navier.vely.v;
                let temp = &navier.temp.v;
                l2_norm(&velx, &velx, &vely, &vely, &temp, &temp, beta1, beta2)
            };

            // Iterate and find optimal initial field
            let mut j_old = 0.;
            let mut alpha = alpha_0;
            for i in 0..max_iter {
                println!("***** Iteration: {:?} ******", i);
                navier.time = 0.;
                navier.pres.v *= 0.;
                navier.pseu.v *= 0.;
                navier.pres.vhat *= 0.;
                navier.pseu.vhat *= 0.;

                // initial fields
                let u0 = navier.velx.v.to_owned();
                let v0 = navier.vely.v.to_owned();
                let t0 = navier.temp.v.to_owned();
                let e0 = l2_norm(&u0, &u0, &v0, &v0, &t0, &t0, beta1, beta2);
                println!("Energy e0: {:10.2e}", e0);

                // integrate_adjoint(&mut navier, 5., Some(1.0));
                let (j, (mut grad_velx, mut grad_vely, mut grad_temp)) =
                    navier.grad_adjoint(max_time, Some(max_time), beta1, beta2, target);

                let decrease_alpha = if MAXIMIZE { j < j_old } else { j > j_old };
                if decrease_alpha && i > 0 {
                    alpha /= 2.;
                    println!("Set alpha: {:4.2e}", alpha);
                    if alpha < 1e-3 {
                        println!("alpha too small. Reset",);
                        alpha = alpha_0;
                        // break;
                    }
                }

                j_old = j;
                funval = j / norm;
                println!("FunVal {:5.3e}", funval);

                // Update fields
                steepest_descent_energy_constrained(
                    &u0,
                    &v0,
                    &t0,
                    &mut grad_velx.v,
                    &mut grad_vely.v,
                    &mut grad_temp.v,
                    &mut navier.velx.v,
                    &mut navier.vely.v,
                    &mut navier.temp.v,
                    beta1,
                    beta2,
                    alpha,
                );
                navier.velx.forward();
                navier.vely.forward();
                navier.temp.forward();
            }

            // Write
            navier.write(&flofile).expect("Unable to write");

            // Write
            let data = format!(
                "{:8.6e}  {:8.6e}  {:8.6e} \n",
                max_time, energy_constraint, funval
            );
            std::fs::write(txtfile, data).expect("Unable to write file");
        }
    }
}
