# rustpde

## `rustpde`: Spectral method solver for Navier-Stokes equations
<img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">

## Dependencies
- cargo >= v1.49
- `hdf5` (sudo apt-get install -y libhdf5-dev)
- `clang` (only for parallel simulations, see [MPI](#mpi).)

This version of `rustpde` contains serial and
mpi-parallel examples of fluid simulations using the spectral method.

## `MPI`

The mpi crate relies on a installed version of libclang. Also
make sure to add the clang bin folder to the path variable, i.e.
for example

- `export PATH="${INSTALL_DIR}/llvm-project/build/bin:$PATH"`

The correct mpi installation can be tricky at times. If you want
to use this library without mpi, you can disable of the default `mpi` feature.
Note that, if default features are turned off, do not forget to
specify which openblas backend you want to use. For example:

- `cargo build --release --no-default-features --features openblas-static`

## `OpenBlas`

By default `rustpde` uses ndarray's `openblas-static` backend,
which is costly for compilation. To use a systems `OpenBlas`
installation, disable default features, and use the `openblas-system`
feature. Make sure to not forget to explicity use the `mpi` feature
in this case, .i.e.
- `cargo build --release --no-default-features --features mpi`

Make sure the `OpenBlas` library is linked correctly in the library path,
i.e.

- `export LIBRARY_PATH="${INSTALL_DIR}/OpenBLAS/lib"`

Openblas multithreading conflicts with internal multithreading.
Turn it off for better performance:
- `export OPENBLAS_NUM_THREADS=1`

## `Hdf5`
Install Hdf5 and link as follows:

- `export HDF5_DIR="${INSTALL_DIR}/hdf5-xx/" `
- ` export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${HDF5_DIR}/lib"`

## Details

Currently `rustpde` implements transforms from physical to spectral space
for the following basis functions:
- `Chebyshev` (Orthonormal), see [`bases::chebyshev()`]
- `ChebDirichlet` (Composite), see [`bases::cheb_dirichlet()`]
- `ChebNeumann` (Composite), see [`bases::cheb_neumann()`]
- `ChebDirichletNeumann` (Composite), see [`bases::cheb_dirichlet_neumann()`]
- `FourierR2c` (Orthonormal), see [`bases::fourier_r2c()`]
- `FourierC2c` (Orthonormal), see [`bases::fourier_c2c()`]

Composite basis combine several basis functions of its parent space to
satisfy the boundary conditions, i.e. Galerkin method.

### Implemented solver

- `2-D Rayleigh Benard Convection: Direct numerical simulation`,
see [`navier_stokes::Navier2D`] or  [`navier_stokes_mpi::Navier2DMpi`]

## Example
Solve 2-D Rayleigh Benard Convection ( Run with `cargo mpirun --np 2 --bin rustpde` )
```rust
use rustpde::mpi::initialize;
use rustpde::mpi::integrate;
use rustpde::navier_stokes_mpi::Navier2DMpi;

fn main() {
    // mpi
    let universe = initialize().unwrap();
    // Parameters
    let (nx, ny) = (65, 65);
    let ra = 1e4;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2DMpi::new(&universe, nx, ny, ra, pr, dt, aspect, adiabatic);
    navier.write_intervall = Some(1.0);
    navier.random_disturbance(1e-4);
    integrate(&mut navier, 10., Some(0.1));
}
```
Solve 2-D Rayleigh Benard Convection with periodic sidewall
```rust
use rustpde::mpi::initialize;
use rustpde::mpi::integrate;
use rustpde::navier_stokes_mpi::Navier2DMpi;

fn main() {
    // mpi
    let universe = initialize().unwrap();
    // Parameters
    let (nx, ny) = (128, 65);
    let ra = 1e4;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2DMpi::new_periodic(&universe, nx, ny, ra, pr, dt, aspect);
    navier.write_intervall = Some(1.0);
    navier.random_disturbance(1e-4);
    integrate(&mut navier, 10., Some(0.1));
}
```

### Postprocess the output

`rustpde` contains some python scripts for postprocessing.
If you have run the above example and specified
to save snapshots, you will see `hdf5` files in the `data` folder.

Plot a single snapshot via

`python3 python/plot2d.py`

or create an animation

`python3 python/anim2d.py`

#### Paraview

The xmf files, corresponding to the h5 files can be created
by the script

`./bin/create_xmf`.

This script works only for fields from the `Navier2D`
solver with the attributes temp, ux, uy and pres.
The bin folder contains also the full crate `create_xmf`, which
can be adapted for specific usecases.

### Documentation

Download and run:

`cargo doc --open`
