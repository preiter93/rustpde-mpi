//! `Hdf5` functions to read write ndarrays with
//! mpi support, to write in parallel
#![cfg(feature = "mpio")]
use crate::mpi::{AsRaw, Universe};
use num_complex::Complex;

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
pub fn write_mpi<'a, A, S, D, T, Sh>(
    universe: &'a Universe,
    filename: &str,
    dsetname: &str,
    array: &ndarray::ArrayBase<S, D>,
    slice: ndarray::SliceInfo<T, D, D>,
    full_shape: Sh,
) -> hdf5::Result<()>
where
    A: hdf5::H5Type,
    S: ndarray::Data<Elem = A>,
    D: ndarray::Dimension,
    T: AsRef<[ndarray::SliceInfoElem]>,
    Sh: Into<hdf5::Extents>,
{
    use std::convert::TryFrom;

    let full_shape = full_shape.into();
    assert!(
        array.ndim() == full_shape.ndim(),
        "Dimension mismatch of array and full_shape, {} vs. {}",
        array.ndim(),
        full_shape.ndim()
    );

    // Create file with mpi access
    let comm = universe.world().as_raw();
    let file = hdf5::FileBuilder::new()
        .with_fapl(|p| p.mpio(comm, None))
        .append(filename)?;

    //Write dataset
    let dset = match file.dataset(dsetname) {
        // Overwrite
        Ok(dset) => dset,
        // Create new dataset
        std::prelude::v1::Err(..) => file
            .new_dataset::<A>()
            .no_chunk()
            .shape(full_shape)
            .create(dsetname)?,
    };
    dset.write_slice(array, hdf5::Hyperslab::try_from(slice)?)?;
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
pub fn write_mpi_complex<'a, A, S, D, T, Sh>(
    universe: &'a Universe,
    filename: &str,
    dsetname: &str,
    array: &ndarray::ArrayBase<S, D>,
    slice: ndarray::SliceInfo<T, D, D>,
    full_shape: Sh,
) -> hdf5::Result<()>
where
    A: hdf5::H5Type + Copy,
    S: ndarray::Data<Elem = Complex<A>>,
    D: ndarray::Dimension,
    T: AsRef<[ndarray::SliceInfoElem]>,
    Sh: Into<hdf5::Extents> + Copy,
    ndarray::SliceInfo<T, D, D>: Copy,
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
pub fn read_mpi<'a, A, D, T>(
    universe: &'a Universe,
    filename: &str,
    dsetname: &str,
    slice: ndarray::SliceInfo<T, D, D>,
) -> hdf5::Result<ndarray::Array<A, D>>
where
    A: hdf5::H5Type,
    D: ndarray::Dimension,
    T: AsRef<[ndarray::SliceInfoElem]>,
{
    // Open file
    let comm = universe.world().as_raw();
    let file = hdf5::FileBuilder::new()
        .with_fapl(|p| p.mpio(comm, None))
        .open(filename)?;

    //Read dataset
    let data = file.dataset(dsetname)?;
    let y: ndarray::ArrayD<A> = data.read_slice(slice)?;

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
pub fn read_mpi_complex<'a, A, D, T>(
    universe: &'a Universe,
    filename: &str,
    dsetname: &str,
    slice: ndarray::SliceInfo<T, D, D>,
) -> hdf5::Result<ndarray::Array<Complex<A>, D>>
where
    A: hdf5::H5Type + Clone + num_traits::Num,
    D: ndarray::Dimension,
    T: AsRef<[ndarray::SliceInfoElem]>,
    ndarray::SliceInfo<T, D, D>: Copy,
{
    // Read real part
    let name_re = format!("{}_re", dsetname);
    let r = read_mpi::<A, D, T>(universe, filename, &name_re, slice)?;

    // Read imaginary part
    let name_im = format!("{}_im", dsetname);
    let i = read_mpi::<A, D, T>(universe, filename, &name_im, slice)?;

    // Add to array
    let mut array = ndarray::Array::<Complex<A>, D>::zeros(r.raw_dim());
    let Complex { mut re, mut im } = array.view_mut().split_complex();
    re.assign(&r);
    im.assign(&i);
    Ok(array)
}
