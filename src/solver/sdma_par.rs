//! Diagonal matrix solver (Parallel)
use super::diag;
use super::{Solve, SolverScalar};
use ndarray::prelude::*;
use ndarray::{Data, DataMut, RemoveAxis, Zip};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul};

/// Solve single diagonal system with diagonals-offsets:  0
#[derive(Debug, Clone)]
pub struct SdmaPar<T> {
    /// Size of matrix (= size of main diagonal)
    pub n: usize,
    /// Main diagonal
    pub dia: Array1<T>,
}

impl<T: SolverScalar> SdmaPar<T> {
    /// Initialize SdmaPar from matrix.
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

impl<T, A, D> Solve<A, D> for SdmaPar<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
    D: Dimension + RemoveAxis,
{
    fn solve<S1: Data<Elem = A>, S2: Data<Elem = A> + DataMut>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) {
        output.assign(input);
        Zip::from(output.lanes_mut(Axis(axis))).par_for_each(|mut out| {
            self.solve_lane(&mut out);
        });
    }
}
