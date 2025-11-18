use raphtory::prelude::{Graph, NodeViewOps, TimeOps};
use raphtory::{core::entities::VID, prelude::GraphViewOps};
use raphtory::db::graph::views::deletion_graph::PersistentGraph;

use crate::diff::{EdgeOp, SnapshotDiffs, ExtendedEdgeOp};

use std::default;
use std::{collections::HashMap, hash::Hash, ops::DerefMut};

pub trait GraphLike{
    type V;

    fn neighbours(&self, node:&Self::V, time: i64) -> impl Iterator<Item=Self::V>;
}

impl GraphLike for PersistentGraph{
    type V = VID;
    fn neighbours(&self, node:&Self::V, time: i64) -> impl Iterator<Item=Self::V> {
        self.at(time).node(node).unwrap().neighbours().iter().map(|x|x.node)
    }
}

pub enum PartitionType<'a, V>{
    All,
    Subset(&'a [V])
}

pub enum PartitionOutput<V>{
    All(HashMap<V,usize>),
    Subset(Vec<usize>)
}

/// Trait to be implemented by Clustering Algorithms that want to consume snapshot diffs 
pub trait SnapshotClusteringAlg<V> {


    /// Called before any ops at this snapshot (optional)
    fn begin_snapshot(&mut self, time: i64) {}

    /// Apply a batch of edge updates
    fn apply_edge_ops(&mut self, time: i64, ops: &[ExtendedEdgeOp<V>], graph: &impl GraphLike);

    /// Called after all ops are applied at this snapshot
    fn extract_partition(&mut self, time: i64, part_type: PartitionType<V>, graph: &impl GraphLike) -> PartitionOutput<V>;

    /// Convenience: process all diffs and collect partitions per snapshot.
    fn process_diffs(
        &mut self,
        diffs: &SnapshotDiffs<V>,
        graph: &impl GraphLike,
    ) -> Vec<(i64, PartitionOutput<V>)> {
        let mut out = Vec::with_capacity(diffs.snapshot_times.len());
        for (t, diff) in diffs.iter() {
            let time = *t;
            self.begin_snapshot(time);
            self.apply_edge_ops(time, diff, graph);
            let partition = self.extract_partition(time, PartitionType::All, graph);
            out.push((time, partition));
        }
        out
    }

    /// Convenience: process diffs and feed each snapshot partition to a callback.
    fn process_diffs_with_subset(
        &mut self,
        diffs: &SnapshotDiffs<V>,
        graph: &impl GraphLike,
        mut subset: Vec<V>,
    )
    {   
        let mut out = Vec::with_capacity(diffs.snapshot_times.len());
        for (t, diff) in diffs.iter() {
            let time = *t;
            self.begin_snapshot(time);
            self.apply_edge_ops(time, diff, graph);
            let partition = self.extract_partition(time, PartitionType::Subset(&subset), graph);
            out.push((time, partition));
        }
    }
}





#[derive(Default)]
pub struct MyClustering{
    adj: HashMap<VID,HashMap<VID,f64>>,
    partition: HashMap<VID, usize>
}

impl SnapshotClusteringAlg<VID> for MyClustering{

    fn apply_edge_ops(&mut self, _time: i64, ops: &[ExtendedEdgeOp<VID>], _graph: &impl GraphLike) {
        for op in ops{
            match op.edge_op(){ 
                EdgeOp::Update(src,dst, w) => {
                    *self.adj
                        .entry(src)
                        .or_default()
                        .entry(dst)
                        .or_default() += w;
                    *self.adj
                        .entry(dst)
                        .or_default()
                        .entry(src)
                        .or_default() += w;
                },
                EdgeOp::DeleteEdge(src, dst, cur_edge_weight) =>{
                    if let Some(neigh) = self.adj.get_mut(&src){
                        neigh.remove(&dst);
                        if neigh.is_empty(){
                            self.adj.remove(&src);
                        }
                    }
                    if let Some(neigh) = self.adj.get_mut(&dst){
                        neigh.remove(&src);
                        if neigh.is_empty(){
                            self.adj.remove(&dst);
                        }
                    }
                }
            }
        }
        self.partition = self.adj.keys().enumerate().map(|(i,x)| (*x,i)).collect();
    }
    
    fn extract_partition(&mut self, _time: i64, part_type: PartitionType<VID>, _graph: &impl GraphLike) -> PartitionOutput<VID> {
        match part_type{
            PartitionType::All => PartitionOutput::All(self.partition.clone()),
            PartitionType::Subset(items) => {
                PartitionOutput::Subset(
                    items.iter().map(|x| *self.partition.get(x).unwrap()).collect::<Vec<usize>>()
                )
            },
        }
    }
}

impl MyClustering{
    pub fn new() -> Self{
        MyClustering { adj: HashMap::new(), partition: HashMap::new() }
    }
}