mod common;
mod coreset_impls;
mod sampling_impls;
mod tree_impls;

use crate::{
    diff::ExtendedEdgeOp,
    snapshot_clustering::{GraphLike, PartitionOutput, PartitionType, SnapshotClusteringAlg},
};
use faer::{
    ColRef,
    sparse::SparseRowMat,
    traits::{Symbolic, num_traits::Zero},
};
use rayon::prelude::*;
use std::{fmt::Debug, hash::Hash, num::NonZero, sync::Arc};

use crate::alg::common::reinterpret_slice;
use crate::diff::{EdgeOp, NodeOps};

use common::*;
use priority_queue::PriorityQueue;
use rustc_hash::{FxHashMap, FxHashSet};

use dyn_stack::{MemBuffer, MemStack};
use faer::{
    Par,
    col::Col,
    mat::Mat,
    matrix_free::eigen::{PartialEigenParams, partial_eigen, partial_eigen_scratch},
    utils::bound::Idx,
};

use linfa::{
    DatasetBase,
    traits::{Fit, Predict},
};
use linfa_clustering::KMeans;
use ndarray::{Array1, ArrayView2, ShapeBuilder};
use num_complex::Complex;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Default, Debug)]
pub struct TreeData<const ARITY: usize> {
    pub timestamp: Vec<usize>,
    pub volume: Vec<Volume>,
    pub size: Vec<usize>,
    pub f_delta: Vec<FDelta>,
    pub h_b: Vec<HB>,
    pub h_s: Vec<HS>,
}

// #[derive(Debug)]
pub struct DynamicClustering<const ARITY: usize, V> {
    // Map stable unique node Ids to tree indices
    pub node_to_tree_map: FxHashMap<V, TreeIndex>,
    // and the reverse map:
    pub tree_to_node_map: FxHashMap<TreeIndex, V>,

    // degree priority queue
    pub degrees: PriorityQueue<V, NodeDegree>,

    // struct to hold tree data
    pub tree_data: TreeData<ARITY>,

    // sigma shift to set
    pub sigma: Float,

    // For lazy query time updates
    pub timestamp: usize,

    pub update_set: FxHashSet<TreeIndex>,

    pub coreset_size: usize,
    pub sampling_seeds: usize,

    pub num_clusters: usize,
    pub cluster_alg:
        Arc<dyn Fn(&mut SparseRowMat<usize, f64>, usize) -> Vec<usize> + Send + Sync + 'static>,
}

impl<const ARITY: usize, V: std::hash::Hash + Eq + Clone + Copy> DynamicClustering<ARITY, V> {
    pub fn new(
        sigma: Float,
        coreset_size: usize,
        sampling_seeds: usize,
        num_clusters: usize,
        cluster_alg: fn(&mut SparseRowMat<usize, f64>, usize) -> Vec<usize>,
    ) -> Self {
        Self {
            node_to_tree_map: Default::default(),
            tree_to_node_map: Default::default(),
            degrees: Default::default(),
            tree_data: Default::default(),
            sigma,
            timestamp: 0,
            update_set: FxHashSet::default(),
            coreset_size,
            sampling_seeds,
            num_clusters,
            cluster_alg: Arc::new(cluster_alg),
        }
    }
}

impl<const ARITY: usize, V: std::hash::Hash + Eq + Clone + Copy + Send + Sync>
    SnapshotClusteringAlg<V> for DynamicClustering<ARITY, V>
{
    fn apply_edge_ops(&mut self, time: i64, ops: &[ExtendedEdgeOp<V>], graph: &impl GraphLike) {}

    fn apply_node_ops(&mut self, time: i64, ops: &NodeOps<V>, _graph: &impl GraphLike) {
        debug_assert_eq!(ops.created_fresh.0.len(), ops.created_fresh.1.len());

        // process fresh nodes
        self.insert_fresh_nodes(ops);

        // take the update set to avoid double borrowing
        let mut update_set = std::mem::take(&mut self.update_set);
        update_set.clear();
        // process stale nodes
        self.update_stale_nodes(ops, &mut update_set);

        // process deleted nodes
        self.update_deleted_nodes(ops, &mut update_set);

        // trigger rebuilding for stale nodes:

        // The size of the tree must remain the same until after we process all updates
        let n = self.tree_data.size.len();
        self.apply_updates_from_set(&update_set, |other, idx| {
            // update size and volume for stale nodes and deleted nodes
            Self::one_step_recompute(idx, &mut other.tree_data.size);
            Self::one_step_recompute(idx, &mut other.tree_data.volume);
        });

        // process modified nodes seperately. They only need volume updates
        update_set.clear();
        self.update_modified_nodes(ops, &mut update_set);
        // process volume updates for modified nodes
        self.apply_updates_from_set(&update_set, |other, idx| {
            // update volume for modified nodes
            Self::one_step_recompute(idx, &mut other.tree_data.volume);
        });

        // restore the update set
        self.update_set = update_set;
    }

    fn extract_partition(
        &mut self,
        time: i64,
        part_type: PartitionType<V>,
        graph: &(impl GraphLike<V = V> + Sync),
    ) -> PartitionOutput<V> {
        // extract coreset:
        let mut coreset = self
            .extract_coreset(graph, self.coreset_size, self.sampling_seeds, time)
            .unwrap();

        // println!("{}", coreset.nodes.len());
        let mut coreset_graph = self.build_coreset_graph(&coreset, time, graph);

        let coreset_labels = (self.cluster_alg)(&mut coreset_graph, self.num_clusters);

        coreset.coreset_labels = Some(coreset_labels.clone());

        let nodes_to_label = match part_type {
            PartitionType::All => None,
            PartitionType::Subset(nodes) => Some(nodes),
        };

        let (names, labels, _distances) =
            self.rust_label_full_graph(&coreset, self.num_clusters, time, graph, nodes_to_label);

        match part_type {
            PartitionType::All => {
                let result = names.into_iter().zip(labels).collect();
                PartitionOutput::All(result)
            }
            PartitionType::Subset(_) => PartitionOutput::Subset(labels),
        }
    }
}

pub fn cluster(graph: &mut SparseRowMat<usize, f64>, k: usize) -> Vec<usize> {
    let n = graph.ncols();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    // Build M = I - 0.5 * (normalized Laplacian) directly from the adjacency and
    // keep D^{-1/2} for optional embedding scaling.
    let deg_inv_sqrt = build_shifted_normalized_laplacian(graph);

    // Random initial vector v0, normalised.
    let mut rng = StdRng::seed_from_u64(42);
    let mut v0 = Col::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let norm = v0.as_ref().norm_l2();
    if norm > 0.0 {
        for v in v0.as_mut().iter_mut() {
            *v /= norm;
        }
    }

    // We need the k largest eigenpairs of M (which correspond to the smallest of the normalized Laplacian).
    let params = PartialEigenParams::default();

    let par = Par::Rayon(NonZero::new(4).unwrap());
    let scratch = partial_eigen_scratch(graph, k, par, params);
    let mut stack_buf = MemBuffer::new(scratch);
    let stack = MemStack::new(&mut stack_buf);

    let mut eigvecs_cplx = Mat::<Complex<f64>>::zeros(n, k);
    let mut eigvals = vec![Complex::new(0.0, 0.0); k];

    let _info = partial_eigen(
        eigvecs_cplx.as_mut(),
        eigvals.as_mut_slice(),
        graph,
        v0.as_ref(),
        f64::EPSILON * 128.0,
        par,
        stack,
        params,
    );

    // take real part of eigenvectors
    let mut eigvecs = Mat::from_fn(n, k, |i, j| eigvecs_cplx[(i, j)].re);

    // normalise embeddings by deg_inv_sqrt vector returned from the builder
    let deg_inv_sqrt = ColRef::from_slice(deg_inv_sqrt.as_slice());
    for mut col in eigvecs.as_mut().col_iter_mut() {
        faer::zip!(&mut col, &deg_inv_sqrt).for_each(|faer::unzip!(val, &scale)| {
            *val *= scale;
        });
    }

    kmeans_labels(&eigvecs, k)
}

/// Build M = I - 0.5 * (normalized Laplacian) in-place on `mat`,
/// i.e., M = 0.5 * I + 0.5 * D^{-1/2} A D^{-1/2}, computed directly from A.
pub fn build_shifted_normalized_laplacian(mat: &mut SparseRowMat<usize, f64>) -> Vec<f64> {
    let (symbolic, vals) = mat.parts_mut();
    let (nrows, _ncols, row_ptr, _row_nnz, col_idx) = symbolic.parts();

    let mut deg = vec![0.0; nrows];
    let mut deg_inv_sqrt = vec![0.0; nrows];
    for i in 0..nrows {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        for idx in start..end {
            deg[i] += vals[idx];
        }
        if deg[i] > 0.0 {
            deg_inv_sqrt[i] = 1.0 / deg[i].sqrt();
        }
    }

    // First write 0.5 * D^{-1/2} A D^{-1/2} into vals.
    for i in 0..nrows {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        let di: f64 = deg_inv_sqrt[i];
        for idx in start..end {
            let j = col_idx[idx];
            let dj = deg_inv_sqrt[j];
            if di > 0.0 && dj > 0.0 {
                vals[idx] = 0.5 * (vals[idx] * di * dj);
                if i == j {
                    vals[idx] += 0.5;
                }
            } else {
                vals[idx] = if i == j { 0.5 } else { 0.0 };
            }
        }
    }
    deg_inv_sqrt
}

pub fn kmeans_labels(embeddings: &Mat<f64>, k: usize) -> Vec<usize> {
    let nrows = embeddings.nrows();
    let ncols = embeddings.ncols();
    if nrows == 0 || ncols == 0 || k == 0 {
        return Vec::new();
    }

    let k = k.min(nrows);
    // zero-copy view into faer storage using its actual strides
    let emb_ref = embeddings.as_ref();
    let stride_row = emb_ref.row_stride();
    let stride_col = emb_ref.col_stride();
    let shape = (nrows, ncols).strides((stride_row as usize, stride_col as usize));
    let view: ArrayView2<'_, f64> = unsafe { ArrayView2::from_shape_ptr(shape, emb_ref.as_ptr()) };
    let dummy_targets = Array1::from_elem(nrows, ());
    let dataset = DatasetBase::new(view, dummy_targets);
    let model = KMeans::params(k)
        .init_method(linfa_clustering::KMeansInit::KMeansPlusPlus)
        .n_runs(3)
        .max_n_iterations(100)
        .fit(&dataset)
        .expect("k-means fit failed");
    let labels = model.predict(&dataset);
    labels.into_raw_vec_and_offset().0
}
