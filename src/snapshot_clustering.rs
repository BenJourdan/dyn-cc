use raphtory::core::entities::VID;

use crate::diff::{EdgeOp, SnapshotDiffs};

use std::{collections::HashMap, hash::Hash, ops::DerefMut};


/// Trait to be implemented by Clustering Algorithms that want to consume snapshot diffs 
pub trait SnapshotClusteringAlg {

    type Partition;

    /// Called before any ops at this snapshot (optional)
    fn begin_snapshot(&mut self, time: i64) {}

    /// Apply a batch of edge updates
    fn apply_edge_ops(&mut self, time: i64, ops: &[EdgeOp]);

    /// Called after all ops are applied at this snapshot
    fn extract_partition(&mut self, time: i64) -> Self::Partition;

    /// Convenience: process all diffs and collect partitions per snapshot.
    fn process_diffs_collect(
        &mut self,
        diffs: &SnapshotDiffs,
    ) -> Vec<(i64, Self::Partition)> {
        let mut out = Vec::with_capacity(diffs.snapshot_times.len());
        for (t, diff) in diffs.iter() {
            let time = *t;
            self.begin_snapshot(time);
            self.apply_edge_ops(time, diff);
            let partition = self.extract_partition(time);
            out.push((time, partition));
        }
        out
    }

    /// Convenience: process diffs and feed each snapshot partition to a callback.
    fn process_diffs_with<F>(
        &mut self,
        diffs: &SnapshotDiffs,
        mut on_snapshot: F,
    ) where
        F: FnMut(i64, &Self::Partition),
    {
        for (t, diff) in diffs.iter() {
            let time = *t;
            self.begin_snapshot(time);
            self.apply_edge_ops(time, diff);
            let partition = self.extract_partition(time);
            on_snapshot(time, &partition);
        }
    }
}


pub struct MyClustering{
    adj: HashMap<VID,HashMap<VID,f64>>,
    partition: HashMap<VID, usize>
}

impl SnapshotClusteringAlg for MyClustering{
    type Partition = HashMap<VID, usize>;
    
    fn apply_edge_ops(&mut self, time: i64, ops: &[EdgeOp]) {
        for op in ops{
            match *op{
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
                EdgeOp::Delete(src, dst) =>{
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
    
    fn extract_partition(&mut self, time: i64) -> Self::Partition {
        self.partition.clone()
    }
}

impl MyClustering{
    pub fn new() -> Self{
        MyClustering { adj: HashMap::new(), partition: HashMap::new() }
    }
}