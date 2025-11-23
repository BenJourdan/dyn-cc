use std::{collections::{HashMap, HashSet}, fmt::Debug};
use crate::{alg::TreeData, diff::{ExtendedEdgeOp, NodeOps}, snapshot_clustering::{
    GraphLike, PartitionOutput, PartitionType, SnapshotClusteringAlg}};
use raphtory::db::graph::node;
use rayon::prelude::*;


use super::common::*;
use super::DynamicClustering;

pub struct Coreset<V>{
    pub nodes: Vec<V>,
    pub weights: Vec<f64>,
}


// Holds info for coreset construction.
pub struct SamplingInfo{
    pub sigma: Float,
    pub simga_over_x_star_deg: Float,
    pub time_stamp: usize,
    pub x_star_seed_set_volume_inv: Float,
    pub total_contribution_inv: Contribution,
}


impl<const ARITY: usize, V: std::hash::Hash + Eq + Clone + Copy> DynamicClustering<ARITY, V> {

    pub fn build_coreset(
        &mut self,
        graph: &impl GraphLike,

    ) -> Coreset<V>{




        todo!()
    }
    
}