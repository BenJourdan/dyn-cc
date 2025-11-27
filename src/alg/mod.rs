mod common;
mod coreset_impls;
mod sampling_impls;
mod tree_impls;

use crate::{
    diff::ExtendedEdgeOp,
    snapshot_clustering::{GraphLike, PartitionOutput, PartitionType, SnapshotClusteringAlg},
};
use faer::{sparse::SparseRowMat, traits::num_traits::Zero};
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
};
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
        Arc<dyn Fn(&SparseRowMat<usize, f64>, usize) -> Vec<usize> + Send + Sync + 'static>,
}

impl<const ARITY: usize, V: std::hash::Hash + Eq + Clone + Copy> DynamicClustering<ARITY, V> {
    pub fn new(
        sigma: Float,
        coreset_size: usize,
        sampling_seeds: usize,
        num_clusters: usize,
        cluster_alg: fn(&SparseRowMat<usize, f64>, usize) -> Vec<usize>,
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
        let coreset = self
            .extract_coreset(graph, self.coreset_size, self.sampling_seeds, time)
            .unwrap();

        // println!("{}", coreset.nodes.len());
        let coreset_graph = self.build_coreset_graph(&coreset, time, graph);

        let coreset_labels = (self.cluster_alg)(&coreset_graph, self.num_clusters);

        let result = coreset.nodes.into_iter().zip(coreset_labels).collect();

        PartitionOutput::All(result)
        // PartitionOutput::Subset(Vec::new())
    }
}

pub fn cluster(graph: &SparseRowMat<usize, f64>, k: usize) -> Vec<usize> {
    let n = graph.ncols();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    // Random initial vector v0, normalised.
    let mut rng = StdRng::seed_from_u64(42);
    let mut v0 = Col::from_fn(n, |_| rng.random_range(-1.0..1.0));
    let norm = v0.as_ref().norm_l2();
    if norm > 0.0 {
        for v in v0.as_mut().iter_mut() {
            *v /= norm;
        }
    }

    let params = PartialEigenParams::default();
    let par = Par::Rayon(NonZero::new(16).unwrap());
    let scratch = partial_eigen_scratch(graph, k, par, params);
    let mut stack_buf = MemBuffer::new(scratch);
    let mut stack = MemStack::new(&mut stack_buf);

    let mut eigvecs = Mat::zeros(n, k);
    let mut eigvals = vec![Complex::new(0.0, 0.0); k];

    let _info = partial_eigen(
        eigvecs.as_mut(),
        eigvals.as_mut_slice(),
        graph,
        v0.as_ref(),
        f64::EPSILON * 128.0,
        par,
        &mut stack,
        params,
    );

    // Placeholder clustering: assign all nodes to cluster 0.
    vec![eigvals[0].re as usize; n]
}
