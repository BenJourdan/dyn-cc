mod common;
mod tree_impls;

use std::{collections::HashMap, fmt::Debug, hash::Hash};
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

        self.insert_fresh_nodes(ops);
    }

    fn extract_partition(&mut self, time: i64, part_type: PartitionType<V>, graph: &impl GraphLike) -> PartitionOutput<V> {
        PartitionOutput::All(HashMap::new())
    }
}
