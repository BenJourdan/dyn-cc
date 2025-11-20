mod common;
mod tree_impls;

use std::{collections::{HashMap, HashSet}, fmt::Debug, hash::Hash};
use crate::{diff::ExtendedEdgeOp, snapshot_clustering::{
    GraphLike, PartitionOutput, PartitionType, SnapshotClusteringAlg}};
use rayon::prelude::*;

use crate::diff::{EdgeOp, NodeOps};
use crate::alg::common::reinterpret_slice;

use common::*;
use priority_queue::PriorityQueue;



#[derive(Default, Debug)]
pub struct TreeData<const ARITY: usize> {
    pub timestamp: Vec<usize>,
    pub volume: Vec<Volume>,
    pub size: Vec<usize>,
    pub f_delta: Vec<FDelta>,
    pub h_b: Vec<HB>,
    pub h_s: Vec<HS>
}



#[derive(Debug)]
pub struct DynamicClustering<const ARITY: usize, V>{

    // Map stable unique node Ids to tree indices
    pub node_to_tree_map: HashMap<V, TreeIndex>,
    // and the reverse map:
    pub tree_to_node_map: HashMap<TreeIndex,V>,

    // degree priority queue
    pub degrees: PriorityQueue<V, NodeDegree>,

    // struct to hold tree data
    pub tree_data: TreeData<ARITY>,

    // sigma shift to set
    pub sigma: Float,

    pub timestamp: usize,

    pub node_creation_buffer: (Vec<V>, Vec<NodeDegree>),
}



impl <const ARITY: usize, V: std::hash::Hash+Eq+Clone + Copy> SnapshotClusteringAlg<V> for DynamicClustering<ARITY, V>{
    fn apply_edge_ops(&mut self, time: i64, ops: &[ExtendedEdgeOp<V>], graph: &impl GraphLike) {
    }

    fn apply_node_ops(&mut self, _time: i64, ops: &NodeOps<V>, _graph: &impl GraphLike) {
        debug_assert_eq!(ops.created_fresh.0.len(), ops.created_fresh.1.len());

        // process fresh nodes
        self.insert_fresh_nodes(ops);

        let mut update_set = HashSet::new();
        // process stale nodes
        self.update_stale_nodes(ops, &mut update_set);

        // process deleted nodes
        self.update_deleted_nodes(ops, &mut update_set);

        // trigger rebuilding for stale nodes:

        // The size of the tree must remain the same until after we process all updates
        let n = self.tree_data.size.len();
        self.apply_updates_from_set(&update_set, | other,idx|{
            let first_child = other.child_index(idx, 0).0;
            let stop = (first_child + ARITY).min(n);
            // update size and volume for stale nodes and deleted nodes
            other.tree_data.size[idx] = other.tree_data.size[first_child..stop].iter().sum();
            other.tree_data.volume[idx] = other.tree_data.volume[first_child..stop].iter().sum();
        });

        // process modified nodes seperately. They only need volume updates
        update_set.clear();
        self.update_modified_nodes(ops, &mut update_set);
        // process volume updates for modified nodes
        self.apply_updates_from_set(&update_set, | other,idx|{
            let first_child = other.child_index(idx, 0).0;
            let stop = (first_child + ARITY).min(n);
            // update volume for modified nodes
            other.tree_data.volume[idx] = other.tree_data.volume[first_child..stop].iter().sum();
        });

    }

    fn extract_partition(&mut self, time: i64, part_type: PartitionType<V>, graph: &impl GraphLike) -> PartitionOutput<V> {
        PartitionOutput::All(HashMap::new())
    }
}
