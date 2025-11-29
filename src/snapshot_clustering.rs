use itertools::Diff;
use raphtory::prelude::*;

use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use raphtory::{core::entities::VID, prelude::GraphViewOps};

use crate::diff::{EdgeOp, ExtendedEdgeOp, NodeOps, SnapshotDiffs};

use rustc_hash::{FxBuildHasher, FxHashMap};
use std::default;
use std::{hash::Hash, ops::DerefMut};

pub trait GraphLike {
    type V;

    fn process_extended_edge_diff(&mut self, diff:&[ExtendedEdgeOp<Self::V>]);

    fn neighbours(&self, node: &Self::V, time: i64) -> impl Iterator<Item = (Self::V, f64)> + Send;

    fn num_nodes(&self, time: i64) -> usize;
    fn nodes(&self, time: i64) -> Vec<Self::V>;

}

impl GraphLike for PersistentGraph {
    type V = VID;
    fn neighbours(&self, node: &Self::V, time: i64) -> impl Iterator<Item = (Self::V, f64)> + Send {
        let graph_at_time = self.at(time);
        let node_at_time = graph_at_time.node(node);
        if node_at_time.is_none() {
            panic!(
                "neighbours() called for missing node {:?} at time {}",
                node, time
            );
        }
        node_at_time.unwrap().neighbours().iter().map(move |x| {
            let nbr = x.node;
            let w = graph_at_time
                .edge(node, &nbr)
                .and_then(|e| e.properties().get("w"))
                .and_then(|p| p.as_f64())
                .unwrap_or(0.0);
            debug_assert!(
                w != 0.0,
                "neighbours() found zero weight between {:?} and {:?} at time {}",
                node,
                nbr,
                time
            );
            (nbr, w)
        })
    }

    fn num_nodes(&self, time: i64) -> usize {
        self.at(time).count_nodes()
    }

    fn nodes(&self, time: i64) -> Vec<Self::V> {
        self.at(time).nodes().into_iter().map(|x| x.node).collect()
    }
    
    fn process_extended_edge_diff(&mut self, diff:&[ExtendedEdgeOp<Self::V>]) {
        
    }
}


#[derive(Default)]
pub struct DiffGraph{
    graph: FxHashMap<VID,FxHashMap<VID,f64>>,
}

impl DiffGraph{
    pub fn with_capacity(capacity: usize)-> Self{
        Self { graph: FxHashMap::with_capacity_and_hasher(capacity, FxBuildHasher) }
    }
}


impl  GraphLike for DiffGraph{
    type V = VID;
    fn neighbours(&self, node: &Self::V, _time: i64) -> impl Iterator<Item = (Self::V, f64)> + Send {
        
        self.graph.get(node).unwrap().iter().map(|(k,v)|(*k,*v))
    }

    fn num_nodes(&self, _time: i64) -> usize {
        self.graph.len()
    }

    fn nodes(&self, _time: i64) -> Vec<Self::V> {
        self.graph.keys().cloned().collect::<Vec<_>>()
    }
    
    fn process_extended_edge_diff(&mut self, diff:&[ExtendedEdgeOp<Self::V>]) {
        const EPS: f64 = 1e-9;

        // helpers to keep the adjacency symmetric and clear out empty nodes
        let remove_edge = |graph: &mut FxHashMap<VID, FxHashMap<VID, f64>>,
                               u: VID,
                               v: VID| {
            if let Some(neighbors) = graph.get_mut(&u) {
                neighbors.remove(&v);
                if neighbors.is_empty() {
                    graph.remove(&u);
                }
            }
            if let Some(neighbors) = graph.get_mut(&v) {
                neighbors.remove(&u);
                if neighbors.is_empty() {
                    graph.remove(&v);
                }
            }
        };

        let upsert_edge =
            |graph: &mut FxHashMap<VID, FxHashMap<VID, f64>>, u: VID, v: VID, w: f64| {
                graph.entry(u).or_default().insert(v, w);
                graph.entry(v).or_default().insert(u, w);
            };

        for op in diff {
            match *op {
                ExtendedEdgeOp::SrcDstPresentUpdate(s, d, delta)
                | ExtendedEdgeOp::SrcMissingUpdate(s, d, delta)
                | ExtendedEdgeOp::DstMissingUpdate(s, d, delta)
                | ExtendedEdgeOp::BothMissingUpdate(s, d, delta) => {
                    let current = self
                        .graph
                        .get(&s)
                        .and_then(|nbrs| nbrs.get(&d))
                        .copied()
                        .unwrap_or(0.0);
                    let new_weight = current + delta;
                    if new_weight.abs() < EPS {
                        remove_edge(&mut self.graph, s, d);
                    } else {
                        upsert_edge(&mut self.graph, s, d, new_weight);
                    }
                }
                ExtendedEdgeOp::SrcDstPresentDelete(s, d, _)
                | ExtendedEdgeOp::SrcRemoveDelete(s, d, _)
                | ExtendedEdgeOp::DstRemoveDelete(s, d, _)
                | ExtendedEdgeOp::BothRemoveDelete(s, d, _) => {
                    remove_edge(&mut self.graph, s, d);
                }
            }
        }
    }
}



pub enum PartitionType<'a, V> {
    All,
    Subset(&'a [V]),
}

#[derive(Debug)]
pub enum PartitionOutput<V> {
    All(FxHashMap<V, usize>),
    Subset(Vec<usize>),
}

/// Trait to be implemented by Clustering Algorithms that want to consume snapshot diffs
pub trait SnapshotClusteringAlg<V> {
    /// Called before any ops at this snapshot (optional)
    fn begin_snapshot(&mut self, time: i64, diffs: &SnapshotDiffs<V>, graph: &(impl GraphLike<V = V> + Sync),) {}


    fn apply_node_ops(&mut self, time: i64, ops: &NodeOps<V>, graph: &impl GraphLike<V = V>);

    /// Called after all ops are applied at this snapshot
    fn extract_partition(
        &mut self,
        time: i64,
        part_type: PartitionType<V>,
        graph: &(impl GraphLike<V = V> + Sync),
    ) -> PartitionOutput<V>;

    fn process_node_diffs_with_subset(
        &mut self,
        diffs: &SnapshotDiffs<V>,
        graph: &mut (impl GraphLike<V = V> + Sync),
        subset: &[V],
    ) -> Vec<(i64, PartitionOutput<V>)> {
        let mut out = Vec::with_capacity(diffs.snapshot_times.len());
        for ((t, node_diff), (_,edge_diff)) in diffs.iter_node_diffs().zip(diffs.iter_edge_diffs()) {
            let time = *t;
            self.begin_snapshot(time, diffs, graph);
            graph.process_extended_edge_diff(edge_diff);
            self.apply_node_ops(time, node_diff, graph);
            let partition = self.extract_partition(time, PartitionType::Subset(subset), graph);
            out.push((time, partition));
        }
        out
    }


}

#[derive(Default)]
pub struct MyClustering {
    adj: FxHashMap<VID, FxHashMap<VID, f64>>,
    partition: FxHashMap<VID, usize>,
}

impl SnapshotClusteringAlg<VID> for MyClustering {
    fn apply_node_ops(&mut self, time: i64, ops: &NodeOps<VID>, graph: &impl GraphLike) {}


    fn extract_partition(
        &mut self,
        _time: i64,
        part_type: PartitionType<VID>,
        _graph: &impl GraphLike,
    ) -> PartitionOutput<VID> {
        match part_type {
            PartitionType::All => PartitionOutput::All(self.partition.clone()),
            PartitionType::Subset(items) => PartitionOutput::Subset(
                items
                    .iter()
                    .map(|x| *self.partition.get(x).unwrap())
                    .collect::<Vec<usize>>(),
            ),
        }
    }
}

impl MyClustering {
    pub fn new() -> Self {
        MyClustering {
            adj: FxHashMap::default(),
            partition: FxHashMap::default(),
        }
    }
}
