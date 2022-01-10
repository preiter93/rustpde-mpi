//! Diagonal matrix solver
use super::diag;
use super::{Solve, SolverScalar};
use ndarray::prelude::*;
use ndarray::{Data, DataMut, RemoveAxis, Zip};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul};

/// Solve single diagonal system with diagonals-offsets:  0
#[derive(Debug, Clone)]
pub struct Sdma<T> {
    /// Size of matrix (= size of main diagonal)
    pub n: usize,
    /// Main diagonal
    pub dia: Array1<T>,
}

impl<T: SolverScalar> Sdma<T> {
    /// Initialize Sdma from matrix.
    /// Extracts the diagonals
    pub fn from_matrix(a: &Array2<T>) -> Self {
        Self {
            n: a.shape()[0],
            dia: diag(a, 0),
        }
    }

    fn solve_lane<A>(&self, input: &mut ArrayViewMut1<A>)
    where
        A: SolverScalar + Div<T, Output = A>,
    {
        self.sdma(input);
    }

    /// b: main-diagonal (0)
    #[allow(clippy::many_single_char_names)]
    fn sdma<A>(&self, d: &mut ArrayViewMut1<A>)
    where
        A: SolverScalar + Div<T, Output = A>,
    {
        let n = d.len();
        let b = self.dia.view();
        for i in 0..n {
            d[i] = d[i] / b[i];
        }
    }
}

impl<T, A, D> Solve<A, D> for Sdma<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
    D: Dimension + RemoveAxis,
{
    /// # Example
    ///```
    /// use rustpde::solver::Sdma;
    /// use rustpde::solver::Solve;
    /// use ndarray::prelude::*;
    /// let nx =  6;
    /// let mut data = Array1::<f64>::zeros(nx);
    /// let mut result = Array1::<f64>::zeros(nx);
    /// let mut matrix = Array2::<f64>::zeros((nx,nx));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// for i in 0..nx {
    ///     let j = (i+1) as f64;
    ///     matrix[[i,i]] = 0.5*j;
    /// }
    /// let solver = Sdma::from_matrix(&matrix);
    /// solver.solve(&data, &mut result,0);
    /// let recover = matrix.dot(&result);
    /// for (a, b) in recover.iter().zip(data.iter()) {
    ///     if (a - b).abs() > 1e-4 {
    ///         panic!("Large difference of values, got {} expected {}.", b, a)
    ///     }
    /// }
    ///```
    fn solve<S1: Data<Elem = A>, S2: Data<Elem = A> + DataMut>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) {
        output.assign(input);
        Zip::from(output.lanes_mut(Axis(axis))).for_each(|mut out| {
            self.solve_lane(&mut out);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use num_complex::Complex;

    fn approx_eq<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        S: Data<Elem = f64>,
        D: Dimension,
    {
        let dif = 1e-3;
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a - b).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    fn approx_eq_complex<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        S: Data<Elem = Complex<f64>>,
        D: Dimension,
    {
        let dif = 1e-3;
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a.re - b.re).abs() > dif || (a.im - b.im).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    #[test]
    fn test_sdma_dim1() {
        let nx = 6;
        let mut data = Array1::<f64>::zeros(nx);
        let mut result = Array1::<f64>::zeros(nx);
        let mut matrix = Array2::<f64>::zeros((nx, nx));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        for i in 0..nx {
            let j = (i + 1) as f64;
            matrix[[i, i]] = 0.5 * j;
        }
        let solver = Sdma::from_matrix(&matrix);
        solver.solve(&data, &mut result, 0);
        let recover: Array1<f64> = matrix.dot(&result);
        approx_eq(&recover, &data);
    }

    #[test]
    fn test_sdma_dim1_complex() {
        let nx = 6;
        let mut data = Array1::<Complex<f64>>::zeros(nx);
        let mut result = Array1::<Complex<f64>>::zeros(nx);
        let mut matrix = Array2::<Complex<f64>>::zeros((nx, nx));
        for (i, v) in data.iter_mut().enumerate() {
            v.re = (i + 0) as f64;
            v.im = (i + 1) as f64;
        }
        for i in 0..nx {
            let j = (i + 1) as f64;
            matrix[[i, i]].re = 0.5 * j;
            matrix[[i, i]].im = 0.5 * j;
        }
        let solver = Sdma::<Complex<f64>>::from_matrix(&matrix);
        solver.solve(&data, &mut result, 0);
        let recover: Array1<Complex<f64>> = matrix.dot(&result);
        approx_eq_complex(&recover, &data);
    }
}
