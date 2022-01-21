//! MPI Routines
pub use funspace::mpi::all_gather_sum;
pub use funspace::mpi::broadcast_scalar;
pub use funspace::mpi::gather_sum;
pub use funspace::mpi::initialize;
pub use funspace::mpi::space_traits::BaseSpaceMpi;
pub use funspace::mpi::Communicator;
pub use funspace::mpi::Decomp2d;
pub use funspace::mpi::Equivalence;
pub use funspace::mpi::Space2 as Space2Mpi;
pub use funspace::mpi::Universe;

const MAX_TIMESTEP: usize = 10_000_000;

/// Integrate trait, step forward in time, and write results
pub trait Integrate {
    /// Update solution
    fn update(&mut self);
    /// Receive current time
    fn get_time(&self) -> f64;
    /// Get timestep
    fn get_dt(&self) -> f64;
    /// Callback function (can be used for i/o)
    fn callback(&mut self);
    /// Additional break criteria
    fn exit(&mut self) -> bool;
    /// Get processor rank
    fn nrank(&self) -> usize;
}

/// Integrade pde, that implements the Integrate trait.
///
/// Specify `save_intervall` to force writing an output.
///
/// Stop Criteria:
/// 1. Timestep limit
/// 2. Time limit
pub fn integrate<T: Integrate>(pde: &mut T, max_time: f64, save_intervall: Option<f64>) {
    let mut timestep: usize = 0;
    let eps_dt = pde.get_dt() * 1e-4;
    loop {
        // Update
        pde.update();
        timestep += 1;

        // Save
        if let Some(dt_save) = &save_intervall {
            if (pde.get_time() % dt_save) < pde.get_dt() / 2.
                || (pde.get_time() % dt_save) > dt_save - pde.get_dt() / 2.
            {
                pde.callback();
            }
        }

        // Break
        if pde.get_time() + eps_dt >= max_time {
            if pde.nrank() == 0 {
                println!("time limit reached: {:?}", pde.get_time());
            }
            break;
        }
        if timestep >= MAX_TIMESTEP {
            if pde.nrank() == 0 {
                println!("timestep limit reached: {:?}", timestep);
            }
            break;
        }
        if pde.exit() {
            if pde.nrank() == 0 {
                println!("break criteria triggered");
            }
            break;
        }
    }
}
