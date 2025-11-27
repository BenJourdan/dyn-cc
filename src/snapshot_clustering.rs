use raphtory::prelude::*;

use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use raphtory::{core::entities::VID, prelude::GraphViewOps};

use crate::diff::{EdgeOp, ExtendedEdgeOp, NodeOps, SnapshotDiffs};

use rustc_hash::FxHashMap;
use std::default;
use std::{hash::Hash, ops::DerefMut};

pub trait GraphLike {
    type V;

    fn neighbours(&self, node: &Self::V, time: i64) -> impl Iterator<Item = (Self::V, f64)> + Send;
}

impl GraphLike for PersistentGraph {
    type V = VID;
    fn neighbours(&self, node: &Self::V, time: i64) -> impl Iterator<Item = (Self::V, f64)> + Send {
        let graph_at_time = self.at(time);
        if graph_at_time.node(node).is_none() {
            panic!(
                "neighbours() called for missing node {:?} at time {}",
                node, time
            );
        }
        graph_at_time
            .node(node)
            .unwrap()
            .neighbours()
            .iter()
            .map(move |x| {
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
    fn begin_snapshot(&mut self, time: i64) {}

    /// Apply a batch of edge updates
    fn apply_edge_ops(
        &mut self,
        time: i64,
        ops: &[ExtendedEdgeOp<V>],
        graph: &impl GraphLike<V = V>,
    );

    fn apply_node_ops(&mut self, time: i64, ops: &NodeOps<V>, graph: &impl GraphLike<V = V>);

    /// Called after all ops are applied at this snapshot
    fn extract_partition(
        &mut self,
        time: i64,
        part_type: PartitionType<V>,
        graph: &(impl GraphLike<V = V> + Sync),
    ) -> PartitionOutput<V>;

    /// Convenience: process all diffs and collect partitions per snapshot.
    fn process_node_diffs(
        &mut self,
        diffs: &SnapshotDiffs<V>,
        graph: &(impl GraphLike<V = V> + Sync),
    ) -> Vec<(i64, PartitionOutput<V>)> {
        let mut out = Vec::with_capacity(diffs.snapshot_times.len());
        for (t, diff) in diffs.iter_node_diffs() {
            let time = *t;
            self.begin_snapshot(time);
            self.apply_node_ops(time, diff, graph);
            let partition = self.extract_partition(time, PartitionType::All, graph);
            out.push((time, partition));
        }
        out
    }

    fn process_edge_diffs(
        &mut self,
        diffs: &SnapshotDiffs<V>,
        graph: &(impl GraphLike<V = V> + Sync),
    ) -> Vec<(i64, PartitionOutput<V>)> {
        let mut out = Vec::with_capacity(diffs.snapshot_times.len());
        for (t, diff) in diffs.iter_edge_diffs() {
            let time = *t;
            self.begin_snapshot(time);
            self.apply_edge_ops(time, diff, graph);
            let partition = self.extract_partition(time, PartitionType::All, graph);
            out.push((time, partition));
        }
        out
    }

    fn process_node_diffs_with_subset(
        &mut self,
        diffs: &SnapshotDiffs<V>,
        graph: &(impl GraphLike<V = V> + Sync),
        mut subset: Vec<V>,
    ) {
        let mut out = Vec::with_capacity(diffs.snapshot_times.len());
        for (t, diff) in diffs.iter_node_diffs() {
            let time = *t;
            self.begin_snapshot(time);
            self.apply_node_ops(time, diff, graph);
            let partition = self.extract_partition(time, PartitionType::Subset(&subset), graph);
            out.push((time, partition));
        }
    }

    fn process_edge_diffs_with_subset(
        &mut self,
        diffs: &SnapshotDiffs<V>,
        graph: &(impl GraphLike<V = V> + Sync),
        mut subset: Vec<V>,
    ) {
        let mut out = Vec::with_capacity(diffs.snapshot_times.len());
        for (t, diff) in diffs.iter_edge_diffs() {
            let time = *t;
            self.begin_snapshot(time);
            self.apply_edge_ops(time, diff, graph);
            let partition = self.extract_partition(time, PartitionType::Subset(&subset), graph);
            out.push((time, partition));
        }
    }
}

#[derive(Default)]
pub struct MyClustering {
    adj: FxHashMap<VID, FxHashMap<VID, f64>>,
    partition: FxHashMap<VID, usize>,
}

impl SnapshotClusteringAlg<VID> for MyClustering {
    fn apply_node_ops(&mut self, time: i64, ops: &NodeOps<VID>, graph: &impl GraphLike) {}

    fn apply_edge_ops(&mut self, _time: i64, ops: &[ExtendedEdgeOp<VID>], _graph: &impl GraphLike) {
        for op in ops {
            match op.edge_op() {
                EdgeOp::Update(src, dst, w) => {
                    *self.adj.entry(src).or_default().entry(dst).or_default() += w;
                    *self.adj.entry(dst).or_default().entry(src).or_default() += w;
                }
                EdgeOp::DeleteEdge(src, dst, cur_edge_weight) => {
                    if let Some(neigh) = self.adj.get_mut(&src) {
                        neigh.remove(&dst);
                        if neigh.is_empty() {
                            self.adj.remove(&src);
                        }
                    }
                    if let Some(neigh) = self.adj.get_mut(&dst) {
                        neigh.remove(&src);
                        if neigh.is_empty() {
                            self.adj.remove(&dst);
                        }
                    }
                }
            }
        }
        self.partition = self.adj.keys().enumerate().map(|(i, x)| (*x, i)).collect();
    }

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
