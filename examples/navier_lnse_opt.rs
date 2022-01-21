//! Gradient based optimization to find fastest growing mode
//!
//! cargo run --release --example navier_lnse_opt

fn main() {
    use num_complex::Complex;
    use num_traits::identities::Zero;
    use rustpde::integrate;
    use rustpde::navier_stokes_lnse::functions::l2_norm;
    use rustpde::navier_stokes_lnse::lnse_adj_grad::opt_routine;
    use rustpde::navier_stokes_lnse::Navier2DLnse;

    // Opt parameter
    let mut alpha = 1.0;
    let (beta1, beta2) = (0.5, 0.5);
    let energy_constraint = 1e-4;
    let max_time = 10.;
    let max_iter = 10;

    // Navier parameter
    let (nx, ny) = (64, 57);
    let ra = 3e3;
    let pr = 0.1;
    let aspect = 1.;
    let dt = 0.01;
    let mut navier = Navier2DLnse::new_periodic(nx, ny, ra, pr, dt, aspect, "hc");
    navier.init_random(1e-3);
    navier.write_unwrap("base.h5");

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

    // Iterate and find optimal initial field
    let mut j_old = 0.;
    for i in 0..max_iter {
        println!("***** Iteration: {:?} ******", i);
        navier.time = 0.;
        navier.pres.v *= 0.;
        navier.pseu.v *= 0.;
        navier.pres.vhat *= Complex::<f64>::zero();
        navier.pseu.vhat *= Complex::<f64>::zero();

        // initial fields
        let u0 = navier.velx.v.to_owned();
        let v0 = navier.vely.v.to_owned();
        let t0 = navier.temp.v.to_owned();
        let e0 = l2_norm(&u0, &u0, &v0, &v0, &t0, &t0, beta1, beta2);
        println!("Energy e0: {:10.2e}", e0);

        // integrate_adjoint(&mut navier, 5., Some(1.0));
        let (j, (mut grad_velx, mut grad_vely, mut grad_temp)) =
            navier.grad_adjoint(max_time, Some(max_time));

        if j < j_old {
            alpha /= 2.;
            println!("Set alpha: {:4.2e}", alpha);
            if alpha < 1e-4 {
                println!("alpha too small. exit",);
                break;
            }
        }
        j_old = j;
        println!("FunVal {:5.3e}", j / energy_constraint);

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
    println!("Evolve field ...");
    navier.read_unwrap("data/opt_field.h5");
    integrate(&mut navier, 20., Some(1.0));
}
