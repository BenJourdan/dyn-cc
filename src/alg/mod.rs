mod common;
mod coreset_impls;
mod sampling_impls;
mod tree_impls;

use crate::{
    diff::ExtendedEdgeOp,
    snapshot_clustering::{GraphLike, PartitionOutput, PartitionType, SnapshotClusteringAlg},
};
use faer::sparse::SparseRowMat;
use rayon::prelude::*;
use std::{fmt::Debug, hash::Hash};

use crate::alg::common::reinterpret_slice;
use crate::diff::{EdgeOp, NodeOps};

use common::*;
use priority_queue::PriorityQueue;
use rustc_hash::{FxHashMap, FxHashSet};

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
    pub cluster_alg: fn(&SparseRowMat<usize, Float>, usize) -> Vec<usize>,
}

impl<const ARITY: usize, V: std::hash::Hash + Eq + Clone + Copy> DynamicClustering<ARITY, V> {
    pub fn new(
        sigma: Float,
        coreset_size: usize,
        sampling_seeds: usize,
        num_clusters: usize,
        cluster_alg: fn(&SparseRowMat<usize, Float>, usize) -> Vec<usize>,
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
            cluster_alg,
        }
    }
}

impl<const ARITY: usize, V: std::hash::Hash + Eq + Clone + Copy + Send + Sync> SnapshotClusteringAlg<V>
    for DynamicClustering<ARITY, V>
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

pub fn cluster(graph: &SparseRowMat<usize, Float>, k: usize) -> Vec<usize> {
    // Simple spectral-ish clustering:
    // - Build dense adjacency from CSR.
    // - Normalize rows by sqrt degree to get features.
    // - Run a basic k-means on those features.

    // let n = graph.ncols();

    // let ones = Mat::ones(n,1);

    // let D = graph * ones;

    // todo!()

    vec![1, 2, 3, 4, 5]
}
