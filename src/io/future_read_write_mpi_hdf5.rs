//! `Hdf5` functions to read write ndarrays with
//! mpi support, to write in parallel
#![cfg(feature = "mpio")]
use crate::mpi::{AsRaw, Universe};
use hdf5::{Error, Extents, FileBuilder, H5Type, Result, Selection};
use ndarray::{Array, ArrayBase, Data, Dimension};
use num_complex::Complex;
use std::convert::TryInto;

/// Write a slice of an ndarray dataset to an
/// hdf5 file. Specify the full shape of the
/// dataset and the slice (with ndarrays `s!` macro).
/// Supplied Array and slices must be smaller than `full_shape`.
///
/// Creates new file or append to existing file.
///
/// # Errors
/// File does not exist or file and dataset exist,
/// but shapes mismatch.
///
/// # Panics
/// Mismatch of number of dimensions between array
/// and `full_shape`.
///
/// # TODO
/// Check that slices do not overlap.
pub fn write_mpi<'a, A, S, D, Sh, Sl>(
    universe: &'a Universe,
    filename: &str,
    dsetname: &str,
    array: &ArrayBase<S, D>,
    slice: Sl,
    full_shape: Sh,
) -> Result<()>
where
    A: H5Type,
    S: Data<Elem = A>,
    D: Dimension,
    Sh: Into<Extents>,
    Sl: TryInto<Selection>,
    Sl::Error: Into<Error>,
{
    let full_shape = full_shape.into();
    assert!(
        array.ndim() == full_shape.ndim(),
        "Dimension mismatch of array and full_shape, {} vs. {}",
        array.ndim(),
        full_shape.ndim()
    );

    // Create file with mpi access
    let comm = universe.world().as_raw();
    // let file = FileBuilder::new()
    //     .with_fapl(|p| p.mpio(comm, None))
    //     .create(filename)?;
    let file = FileBuilder::new().with_fapl(|p| p.mpio(comm, None)).append(filename)?;

    // //Write dataset
    // let dset = file
    //     .new_dataset::<A>()
    //     .shape(full_shape)
    //     .create(dsetname)
    //     .expect("Failed to create dataset.");
    // It is better to use `unwrap_or_else` which is lazily evaluated
    // than `unwrap_or`.
       let dset = file.dataset(dsetname).unwrap_or_else(|_| {
           file.new_dataset::<A>()
               //.no_chunk()
               .shape(full_shape)
               .create(dsetname)
               .expect("Failed to create dataset.")
       });

    dset.write_slice(array, slice.try_into().map_err(|err| err.into())?)?;
    Ok(())
}

/// Write a slice of an ndarray dataset to an
/// hdf5 file. Specify the full shape of the
/// dataset and the slice (with ndarrays `s!` macro).
/// Supplied Array and slices must be smaller than `full_shape`.
///
/// Creates new file or append to existing file.
///
/// # Errors
/// File does not exist or file and dataset exist,
/// but shapes mismatch.
///
/// # Panics
/// Mismatch of number of dimensions between array
/// and `full_shape`.
pub fn write_mpi_complex<'a, A, S, D, Sh, Sl>(
    universe: &'a Universe,
    filename: &str,
    dsetname: &str,
    array: &ArrayBase<S, D>,
    slice: Sl,
    full_shape: Sh,
) -> Result<()>
where
    A: H5Type + Copy,
    S: Data<Elem = Complex<A>>,
    D: Dimension,
    Sh: Into<Extents> + Copy,
    Sl: TryInto<Selection> + Copy,
    Sl::Error: Into<Error>,
{
    // Write real part
    let name_re = format!("{}_re", dsetname);
    write_mpi(
        universe,
        filename,
        &name_re,
        &array.map(|x| x.re),
        slice,
        full_shape,
    )?;
    // Write imag part
    let name_im = format!("{}_im", dsetname);
    write_mpi(
        universe,
        filename,
        &name_im,
        &array.map(|x| x.im),
        slice,
        full_shape,
    )?;
    Ok(())
}

/// Read slice of an ndarray dataset
///
/// # Errors
/// File or dataset do not exist.
///
/// # Panics
/// Panics when `into_dimensionality` fails.
///
/// # TODO
/// Check that slices do not overlap.
pub fn read_mpi<'a, A, D, S>(
    universe: &'a Universe,
    filename: &str,
    dsetname: &str,
    slice: S,
    // slice: ndarray::SliceInfo<T, D, D>,
) -> Result<Array<A, D>>
where
    A: H5Type,
    D: Dimension,
    //T: AsRef<[ndarray::SliceInfoElem]>,
    S: TryInto<Selection>,
    S::Error: Into<Error>,
{
    // Open file
    let comm = universe.world().as_raw();
    let file = hdf5::FileBuilder::new()
        .with_fapl(|p| p.mpio(comm, None))
        .open(filename)?;

    //Read dataset
    let data = file.dataset(dsetname)?;
    let y: ndarray::ArrayD<A> = data.read_slice(slice.try_into().map_err(|err| err.into())?)?;

    // Dyn to static
    let x = y.into_dimensionality::<D>().unwrap();
    Ok(x)
}

/// Read slice of an ndarray dataset
///
/// # Errors
/// File or dataset do not exist.
///
/// # Panics
/// Panics when `into_dimensionality` fails.
///
/// # TODO
/// Check that slices do not overlap.
pub fn read_mpi_complex<'a, A, D, S>(
    universe: &'a Universe,
    filename: &str,
    dsetname: &str,
    slice: S,
) -> Result<Array<Complex<A>, D>>
where
    A: H5Type + Clone + num_traits::Num,
    D: Dimension,
    S: TryInto<Selection> + Copy,
    S::Error: Into<Error>,
{
    // Read real part
    let name_re = format!("{}_re", dsetname);
    let r = read_mpi::<A, D, S>(universe, filename, &name_re, slice)?;

    // Read imaginary part
    let name_im = format!("{}_im", dsetname);
    let i = read_mpi::<A, D, S>(universe, filename, &name_im, slice)?;

    // Add to array
    let mut array = Array::<Complex<A>, D>::zeros(r.raw_dim());
    let Complex { mut re, mut im } = array.view_mut().split_complex();
    re.assign(&r);
    im.assign(&i);
    Ok(array)
}
