//! Run example:
//!
//! cargo mpirun --np 2 --bin rustpde --release
//!
//! Important: Disable obenblas multithreading:
//! ```
//! export OPENBLAS_NUM_THREADS=1
//! ```
//! Gradient based optimization to find fastest growing mode
//!
//! cargo run --release --example navier_lnse_opt
//
fn main() {
    use num_complex::Complex;
    use num_traits::identities::Zero;
    use rustpde::integrate;
    use rustpde::navier_stokes::Navier2D;
    use rustpde::navier_stokes_lnse::functions::l2_norm;
    use rustpde::navier_stokes_lnse::lnse_adj_grad::opt_routine;
    use rustpde::navier_stokes_lnse::lnse_adj_grad::MAXIMIZE;
    use rustpde::navier_stokes_lnse::Navier2DLnse;
    use rustpde::Integrate;

    // Opt parameter
    let mut alpha = 1.0;
    let (beta1, beta2) = (0.5, 0.5);
    let energy_constraint = 3e1;
    let max_time = 15.;
    let max_iter = 12;

    // Navier parameter
    let (nx, ny) = (128, 57);
    let ra = 1e5;
    let pr = 1.;
    let aspect = 1.;
    let dt = 0.05;
    let mut navier_nl = Navier2D::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
    let mut navier = Navier2DLnse::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
    navier.init_random(1e-3);
    // navier.write_unwrap("base.h5");
    // navier.read_unwrap("data/opt_field.h5");

    // Test with target state
    let mut target = navier.mean.clone();
    // // a
    // target
    //     .velx
    //     .v
    //     .assign(&(-2. * &target.velx.v.to_owned().slice(ndarray::s![..;-1,..])));
    // target.vely.v.assign(&(-2. * target.vely.v.to_owned()));
    // target.temp.v.assign(
    //     &(2. * &target.temp.v.to_owned().slice(ndarray::s![..;-1,..])
    //         - &navier_nl.tempbc.as_ref().unwrap().v),
    // );
    // b
    target
        .velx
        .v
        .assign(&(-1. * &target.velx.v.to_owned().slice(ndarray::s![..;-1,..])));
    target.vely.v.assign(&(-1. * target.vely.v.to_owned()));
    target.temp.v.assign(
        &(1. * &target.temp.v.to_owned().slice(ndarray::s![..;-1,..]))
    );
    target.velx.v -= &navier.mean.velx.v;
    target.vely.v -= &navier.mean.vely.v;
    target.temp.v -= &navier.mean.temp.v;
    target.velx.v *= 1.;
    target.vely.v *= 1.;
    target.temp.v *= 1.;

    //
    target.velx.forward();
    target.vely.forward();
    target.temp.forward();
    target.write_unwrap("data/target.h5");

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

    let target = Some(&target);

    // Normalize constant for later
    let norm = if let Some(t) = &target {
        let velx = &(&navier.velx.v - &t.velx.v);
        let vely = &(&navier.vely.v - &t.vely.v);
        let temp = &(&navier.temp.v - &t.temp.v);
        l2_norm(&velx, &velx, &vely, &vely, &temp, &temp, beta1, beta2)
    } else {
        let velx = &navier.velx.v ;
        let vely = &navier.vely.v;
        let temp = &navier.temp.v;
        l2_norm(&velx, &velx, &vely, &vely, &temp, &temp, beta1, beta2)
    };

    // Iterate and find optimal initial field
    let mut j_old = 0.;
    for i in 0..max_iter {
        println!("***** Iteration: {:?} ******", i);
        navier.time = 0.;
        navier.pres.v *= 0.;
        navier.pseu.v *= 0.;
        // navier.pres.vhat *= Complex::<f64>::zero();
        // navier.pseu.vhat *= Complex::<f64>::zero();
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
        // Some(&target)

        let decrease_alpha = if MAXIMIZE { j < j_old } else { j > j_old };
        if decrease_alpha && i > 0 {
            alpha /= 2.;
            println!("Set alpha: {:4.2e}", alpha);
            if alpha < 1e-4 {
                println!("alpha too small. exit",);
                break;
            }
        }

        j_old = j;
        println!("FunVal {:5.3e}", j / norm);

        // Update fields
        opt_routine(
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
    navier
        .write("data/opt_field.h5")
        .expect("Unable to write opt_field.h5");

    // Finally evolve field
    // println!("Evolve field ...");
    // navier.read_unwrap("data/opt_field.h5");
    // navier.reset_time();
    // integrate(&mut navier, 20., Some(1.0));

    // Non linear simulation
    navier_nl
        .velx
        .v
        .assign(&(&navier.velx.v + &navier.mean.velx.v));
    navier_nl
        .vely
        .v
        .assign(&(&navier.vely.v + &navier.mean.vely.v));
    if let Some(field) = &navier_nl.tempbc {
        navier_nl
            .temp
            .v
            .assign(&(&navier.temp.v + &navier.mean.temp.v - &field.v));
    } else {
        navier_nl
            .temp
            .v
            .assign(&(&navier.temp.v + &navier.mean.temp.v));
    }
    navier_nl.velx.forward();
    navier_nl.vely.forward();
    navier_nl.temp.forward();
    navier_nl.callback();
    integrate(&mut navier_nl, 40., Some(0.5));

    // Final energy
    let final_energy = if let Some(t) = &target {
        let velx = &(&navier_nl.velx.v - &t.velx.v);
        let vely = &(&navier_nl.vely.v - &t.vely.v);
        let temp = &(&navier_nl.temp.v - &t.temp.v);
        l2_norm(&velx, &velx, &vely, &vely, &temp, &temp, beta1, beta2)
    } else {
        let velx = &navier.velx.v ;
        let vely = &navier.vely.v;
        let temp = &navier.temp.v;
        l2_norm(&velx, &velx, &vely, &vely, &temp, &temp, beta1, beta2)
    };
    println!("Final energy {:?}", final_energy);
}

fn main3() {
    use rustpde::integrate;
    use rustpde::navier_stokes::Navier2D;
    // Parameters
    let (nx, ny) = (128, 57);
    let ra = 1e5;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.05;
    let mut navier = Navier2D::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
    // if navier.nrank() == 0 {
    //     navier.write("restart.h5");
    // }
    //navier.read_unwrap("restart.h5");

    // Set initial conditions
    navier.random_disturbance(1e-2);
    // navier.set_temperature(1e-1, 2., 1.);
    // navier.set_velocity(1e-1, 2., 1.);
    integrate(&mut navier, 300., Some(1.0));
}
