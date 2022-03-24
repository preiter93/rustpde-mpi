//! Implementations of volumetric weight averages
use super::{BaseSpace, BaseSpaceMpi, FieldBaseMpi};
use crate::mpi::Equivalence;
use crate::types::FloatNum;
use funspace::mpi::all_gather_sum;
use ndarray::{s, Array1, ArrayBase, Axis, Data, Ix1};

impl<A, T2, S> FieldBaseMpi<A, A, T2, S, 2>
where
    A: FloatNum + Equivalence + std::iter::Sum,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2> + BaseSpaceMpi<A, 2>,
{
    /// Return volumetric weighted average along lane
    pub fn average_lane_mpi<S1: Data<Elem = A>>(
        &self,
        lane: &ArrayBase<S1, Ix1>,
        dx: &ArrayBase<S1, Ix1>,
        length: A,
        is_contig: bool,
    ) -> A {
        // Average
        let average = (lane * dx / length).sum();
        // Check if domain is mpi broken
        if is_contig {
            // Axis is contiguous
            average
        } else {
            // Axis is non contiguous
            let mut average_global = A::zero();
            all_gather_sum(self.space.get_universe(), &average, &mut average_global);
            average_global
        }
    }

    /// Return volumetric weighted average along axis
    ///
    /// # Panics
    /// If the wrong `DecompHandler` was returned,
    /// which does not match the array size
    pub fn average_axis_mpi(&self, axis: usize) -> Array1<A> {
        // Get mpi decomp
        let space = &self.space;
        let dcp = &space
            .get_decomp_from_global_shape(&space.shape_physical())
            .y_pencil;
        assert!(dcp.sz == self.v_y_pen.shape());
        let length = self.dx[axis].sum();
        let dx = self.dx[axis].slice(s![dcp.st[axis]..=dcp.en[axis]]);
        let is_contig = axis == dcp.axis_contig;
        let mut avg = Array1::<A>::zeros([self.v_y_pen.shape()[1], self.v_y_pen.shape()[0]][axis]);
        for (lane, x) in self
            .v_y_pen
            .lanes(Axis(axis))
            .into_iter()
            .zip(avg.iter_mut())
        {
            *x = self.average_lane_mpi(&lane, &dx, length, is_contig);
        }
        avg
    }

    /// Return volumetric weighted average
    pub fn average_mpi(&self) -> A {
        // Get mpi decomp
        let space = &self.space;
        let dcp = &space
            .get_decomp_from_global_shape(&space.shape_physical())
            .y_pencil;
        // Average x
        let dx = self.dx[1].slice(s![dcp.st[1]..=dcp.en[1]]);
        let length = self.dx[1].sum();
        let mut avg_x = Array1::<A>::zeros(dx.raw_dim());
        avg_x.assign(&(self.average_axis_mpi(0) * dx / length));
        // Average y
        let avg = avg_x.sum_axis(Axis(0));
        // Check if domain is mpi broken
        if dcp.axis_contig == 1 {
            avg[[]]
        } else {
            let mut avg_global = A::zero();
            all_gather_sum(self.space.get_universe(), &avg[[]], &mut avg_global);
            avg_global
        }
    }
}
