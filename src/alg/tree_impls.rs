use crate::{
    alg::{TreeData, coreset_impls::SamplingInfo},
    diff::{ExtendedEdgeOp, NodeOps},
    snapshot_clustering::{GraphLike, PartitionOutput, PartitionType, SnapshotClusteringAlg},
};
use raphtory::db::graph::node;
use rayon::prelude::*;
use std::fmt::Debug;

use super::DynamicClustering;
use super::common::*;
use rustc_hash::FxHashSet;

impl<const ARITY: usize, V: std::hash::Hash + Eq + Clone + Copy> DynamicClustering<ARITY, V> {
    pub fn parent_index(&self, child_index: TreeIndex) -> Option<TreeIndex> {
        if child_index.0 == 0 {
            None
        } else {
            Some((child_index - TreeIndex(1)) / ARITY)
        }
    }

    pub fn child_index(&self, parent_index: TreeIndex, child_index: usize) -> TreeIndex {
        TreeIndex(parent_index.0 * ARITY + 1 + child_index)
    }

    pub fn which_child(child_idex: TreeIndex) -> usize {
        (child_idex.0 - 1) % ARITY
    }

    pub fn num_leaves(&self) -> usize {
        let n = self.node_to_tree_map.len();
        debug_assert!(
            self.tree_to_node_map.len() == n && self.degrees.len() == n,
            "Inconsistent number of nodes in DynamicClustering data structures"
        );
        n
    }

    pub fn num_internal_nodes(&self) -> usize {
        if self.num_leaves() <= 1 {
            0
        } else {
            (self.num_leaves() - 2) / (ARITY - 1) + 1
        }
    }

    pub fn num_internal_nodes_from_leaves(num_leaves: usize) -> usize {
        if num_leaves <= 1 {
            0
        } else {
            (num_leaves - 2) / (ARITY - 1) + 1
        }
    }

    pub fn num_total_nodes(&self) -> usize {
        self.num_leaves() + self.num_internal_nodes()
    }

    pub fn num_total_nodes_from_leaves(num_leaves: usize) -> usize {
        num_leaves + Self::num_internal_nodes_from_leaves(num_leaves)
    }

    #[inline]
    pub fn assert_zero_volume_for_empty_leaves(&self, info: &SamplingInfo<V>) {
        let leaf_start = self.num_internal_nodes();
        for (i, (sz, vol)) in self.tree_data.size[leaf_start..]
            .iter()
            .zip(self.tree_data.volume[leaf_start..].iter())
            .enumerate()
        {
            if *sz == 0 {
                let f_val = self.f(TreeIndex(leaf_start + i), info);
                assert!(
                    vol.0 == 0.0,
                    "leaf {} has size 0 but non-zero volume {}",
                    leaf_start + i,
                    vol.0
                );
                assert!(
                    f_val.0 == 0.0,
                    "leaf {} has size 0 but non-zero contribution {}",
                    leaf_start + i,
                    f_val
                );
            }
        }
    }

    pub fn insert_fresh_nodes(&mut self, node_ops: &NodeOps<V>) {
        let fresh_nodes = node_ops.created_fresh.0.as_slice();
        let fresh_node_degrees =
            reinterpret_slice::<f64, Volume>(node_ops.created_fresh.1.as_slice());
        debug_assert_eq!(
            fresh_nodes.len(),
            fresh_node_degrees.len(),
            "new_nodes and new_node_degrees must have the same length"
        );
        debug_assert!(ARITY > 1, "ARITY must be at least 2");

        let added = fresh_nodes.len();
        if added == 0 {
            return;
        }

        let old_leaves = self.num_leaves();
        let I0 = self.num_internal_nodes();
        let old_total = I0 + old_leaves;

        let new_leaves = old_leaves + added;
        let I1 = Self::num_internal_nodes_from_leaves(new_leaves);
        let new_total = new_leaves + I1;

        // grow buffers
        self.tree_data.timestamp.resize(new_total, 0);
        self.tree_data.volume.resize(new_total, Volume::zero());
        self.tree_data.size.resize(new_total, 1);
        self.tree_data.f_delta.resize(new_total, FDelta::zero());
        self.tree_data.h_b.resize(new_total, HB::zero());
        self.tree_data.h_s.resize(new_total, HS::zero());

        let old_leaf_start = I0;
        let old_leaf_end = I0 + old_leaves;

        if I1 >= old_total {
            // --- Big height jump: move *all* old leaves to [I1 .. I1+old_leaves) ---

            let new_leaf_start = I1;

            self.tree_data
                .volume
                .copy_within(old_leaf_start..old_leaf_end, new_leaf_start);

            // fix maps for all old leaves
            for i in 0..old_leaves {
                let old_idx = TreeIndex(old_leaf_start + i);
                let node = self.tree_to_node_map.remove(&old_idx).unwrap();
                let new_idx = TreeIndex(new_leaf_start + i);
                self.tree_to_node_map.insert(new_idx, node);
                self.node_to_tree_map.insert(node, new_idx);
            }

            // append new leaves
            let start_new = new_leaf_start + old_leaves;
            let end_new = start_new + added;

            self.tree_data.volume[start_new..end_new].copy_from_slice(fresh_node_degrees);

            for (i, (&deg, node_ref)) in fresh_node_degrees
                .iter()
                .zip(fresh_nodes.iter())
                .enumerate()
            {
                let node = node_ref;
                let idx = TreeIndex(start_new + i);
                self.tree_to_node_map.insert(idx, *node);
                self.node_to_tree_map.insert(*node, idx);
                self.degrees.push(*node, convert(deg));
            }

            // changed leaves = [new_leaf_start .. end_new)
            self.rebuild_from_leaves(new_leaf_start, end_new);
        } else {
            // --- Small step: only move prefix [I0 .. I1) of leaves ---

            let src_start = I0;
            let src_end = I1; // <= old_total here
            let promoted = src_end - src_start; // number of promoted leaves

            let dest_start = old_total;
            let dest_end = dest_start + promoted;

            self.tree_data
                .volume
                .copy_within(src_start..src_end, dest_start);

            // update maps for moved leaves
            for i in 0..promoted {
                let old_idx = TreeIndex(src_start + i);
                let node = self.tree_to_node_map.remove(&old_idx).unwrap();
                let new_idx = TreeIndex(dest_start + i);
                self.tree_to_node_map.insert(new_idx, node);
                self.node_to_tree_map.insert(node, new_idx);
            }

            // leaves in [I1 .. old_total) stay in place; indices already >= I1

            // append new leaves after moved block
            let start_new = dest_end;
            let end_new = start_new + added;
            debug_assert_eq!(end_new, I1 + new_leaves);

            self.tree_data.volume[start_new..end_new].copy_from_slice(fresh_node_degrees);

            for (i, (&deg, node_ref)) in fresh_node_degrees
                .iter()
                .zip(fresh_nodes.iter())
                .enumerate()
            {
                let node = *node_ref;
                let idx = TreeIndex(start_new + i);
                self.tree_to_node_map.insert(idx, node);
                self.node_to_tree_map.insert(node, idx);
                self.degrees.push(node, convert(deg));
            }

            // changed leaves (promoted + new) = [old_total .. end_new)
            self.rebuild_from_leaves(old_total, end_new);
        }

        debug_assert_eq!(self.num_leaves(), new_leaves);
        debug_assert_eq!(self.num_internal_nodes(), I1);
    }

    fn rebuild_from_leaves(&mut self, mut leaf_start: usize, mut leaf_end: usize) {
        // Precondition: [leaf_start, leaf_end) is a contiguous block of leaves
        // (possibly spanning the last 2 levels).

        let total = self.num_total_nodes();

        if leaf_start >= leaf_end || leaf_start == 0 {
            return;
        }

        // --- compute bottom-level start index ---

        let n = total as f64;
        let d = ARITY as f64;

        // For a full d-ary tree:
        // N = (d^(h+1) - 1)/(d-1) -> h = log_d((d-1)N + 1) - 1
        // We floor h here; for a complete tree this gives the deepest *full* level,
        // and the "bottom" level is either h or h+1, but the boundary
        // (first index of deepest level) is still:
        //   l_bottom_start = (d^h - 1)/(d-1)
        let h = (((d - 1.0) * n + 1.0).log(d)).floor() as u32 - 1;
        let l_bottom_start = (ARITY.pow(h) - 1) / (ARITY - 1);

        // Split into:
        //  - bottom_range: indices on the deepest level
        //  - top_range: indices on the level above
        //
        // Either range may be empty.
        let mut bottom_range = leaf_start.max(l_bottom_start)..leaf_end.max(l_bottom_start);
        let mut top_range = leaf_start.min(l_bottom_start)..leaf_end.min(l_bottom_start);

        // Invariants we maintain:
        //  - bottom_range and top_range are each either empty or entirely one level.
        //  - bottom_range (if non-empty) is strictly deeper than top_range (if non-empty).

        while !bottom_range.is_empty() || !top_range.is_empty() {
            // --- 1. process bottom_range (deepest level) ---

            if !bottom_range.is_empty() {
                let child_start = bottom_range.start;
                let child_end = bottom_range.end;

                if child_start == 0 {
                    // We've reached the root.
                    bottom_range = 0..0;
                    continue;
                }

                // Compute parent range for this level
                let parent_start = self.parent_index(TreeIndex(child_start)).unwrap().0;
                let parent_end = self.parent_index(TreeIndex(child_end - 1)).unwrap().0 + 1;

                // Update parents from their children.
                // This loop is *per-level* and parallelisable.
                for p in parent_start..parent_end {
                    let p_idx = TreeIndex(p);
                    let c_start = self.child_index(p_idx, 0).0;
                    let c_end = (c_start + ARITY).min(total);

                    let size: usize = self.tree_data.size[c_start..c_end].iter().sum();
                    let volume: Volume = self.tree_data.volume[c_start..c_end].iter().sum();

                    self.tree_data.size[p] = size;
                    self.tree_data.volume[p] = volume;
                }

                // Now our "bottom" frontier moves up one level
                bottom_range = parent_start..parent_end;
            }

            // --- 2. possibly merge with top_range ---

            if !bottom_range.is_empty() && !top_range.is_empty() {
                // If the new bottom_range overlaps with the existing top_range,
                // they are now on the same level: merge them.
                if bottom_range.end >= top_range.start && bottom_range.start <= top_range.end {
                    let new_start = bottom_range.start.min(top_range.start);
                    let new_end = bottom_range.end.max(top_range.end);
                    top_range = new_start..new_end;
                    bottom_range = 0..0; // empty
                }
            }

            // --- 3. process top_range (next level up) ---

            if !top_range.is_empty() {
                let child_start = top_range.start;
                let child_end = top_range.end;

                if child_start == 0 {
                    // We're at the root: update it directly and finish.
                    let p_idx = TreeIndex(0);
                    let c_start = self.child_index(p_idx, 0).0;
                    let c_end = (c_start + ARITY).min(total);

                    let size: usize = self.tree_data.size[c_start..c_end].iter().sum();
                    let volume: Volume = self.tree_data.volume[c_start..c_end].iter().sum();

                    self.tree_data.size[0] = size;
                    self.tree_data.volume[0] = volume;
                    break;
                }

                let parent_start = self.parent_index(TreeIndex(child_start)).unwrap().0;
                let parent_end = self.parent_index(TreeIndex(child_end - 1)).unwrap().0 + 1;

                for p in parent_start..parent_end {
                    let p_idx = TreeIndex(p);
                    let c_start = self.child_index(p_idx, 0).0;
                    let c_end = (c_start + ARITY).min(total);

                    let size: usize = self.tree_data.size[c_start..c_end].iter().sum();
                    let volume: Volume = self.tree_data.volume[c_start..c_end].iter().sum();

                    self.tree_data.size[p] = size;
                    self.tree_data.volume[p] = volume;
                }

                // Move top frontier up one level
                top_range = parent_start..parent_end;
            }
        }

        // Ensure the root reflects the final child aggregates.
        if total > 1 {
            let root_idx = TreeIndex(0);
            let child_start = self.child_index(root_idx, 0).0;
            if child_start < total {
                let child_end = (child_start + ARITY).min(total);
                let size: usize = self.tree_data.size[child_start..child_end].iter().sum();
                let volume: Volume = self.tree_data.volume[child_start..child_end].iter().sum();
                self.tree_data.size[0] = size;
                self.tree_data.volume[0] = volume;
            }
        }
    }

    pub fn update_stale_nodes(
        &mut self,
        node_ops: &NodeOps<V>,
        update_set: &mut FxHashSet<TreeIndex>,
    ) {
        // insert nodes that have previously been deleted (but not removed from the tree)
        // We add the indices to update to update_set

        let stale_nodes = node_ops.created_stale.0.as_slice();
        let stale_node_degrees =
            reinterpret_slice::<f64, Volume>(node_ops.created_stale.1.as_slice());
        stale_nodes
            .iter()
            .zip(stale_node_degrees)
            .for_each(|(v, d)| {
                let idx = *self.node_to_tree_map.get(v).unwrap();
                debug_assert!(self.tree_data.size[idx] == 0);
                debug_assert!(self.tree_data.volume[idx] == Volume(0.0.into()));
                self.tree_data.size[idx] = 1;
                self.tree_data.volume[idx] = *d;
                update_set.insert(idx);
                // also reinsert into degree queue
                self.degrees.push(*v, convert(*d));
            });
    }

    pub fn update_modified_nodes(
        &mut self,
        node_ops: &NodeOps<V>,
        update_set: &mut FxHashSet<TreeIndex>,
    ) {
        // insert nodes that have previously been deleted (but not removed from the tree)
        // We add the indices to update to update_set

        let modified_nodes = node_ops.modified.0.as_slice();
        let modified_node_degrees =
            reinterpret_slice::<f64, Volume>(node_ops.modified.1.as_slice());
        modified_nodes
            .iter()
            .zip(modified_node_degrees)
            .for_each(|(v, d)| {
                let idx = *self.node_to_tree_map.get(v).unwrap();
                debug_assert!(self.tree_data.size[idx] == 1);
                debug_assert!(self.tree_data.volume[idx] != Volume(0.0.into()));
                self.tree_data.volume[idx] = *d;
                update_set.insert(idx);
                // also update degree queue
                self.degrees.push(*v, convert(*d));
            });
    }

    pub fn update_deleted_nodes(
        &mut self,
        node_ops: &NodeOps<V>,
        update_set: &mut FxHashSet<TreeIndex>,
    ) {
        // insert nodes that have previously been deleted (but not removed from the tree)
        // We add the indices to update to update_set

        let deleted_nodes = node_ops.deleted.0.as_slice();

        deleted_nodes.iter().for_each(|v| {
            let idx = *self.node_to_tree_map.get(v).unwrap();
            debug_assert!(self.tree_data.size[idx] == 1);
            debug_assert!(self.tree_data.volume[idx] != Volume(0.0.into()));
            self.tree_data.size[idx] = 0;
            self.tree_data.volume[idx] = Volume(0.0.into());
            update_set.insert(idx);
            // also remove from degree queue
            self.degrees.push(*v, NodeDegree::zero());
        });
    }

    pub fn apply_updates_from_set<F: Fn(&mut Self, TreeIndex)>(
        &mut self,
        update_set: &FxHashSet<TreeIndex>,
        f: F,
    ) {
        if update_set.is_empty() {
            return;
        }

        let mut current = FxHashSet::default();
        let mut bottom = FxHashSet::default();

        let total = self.num_total_nodes();
        let n = total as f64;
        let d = ARITY as f64;

        // For a full d-ary tree:
        // N = (d^(h+1) - 1)/(d-1) -> h = log_d((d-1)N + 1) - 1
        // We floor h here; for a complete tree this gives the deepest *full* level,
        // and the "bottom" level is either h or h+1, but the boundary
        // (first index of deepest level) is still:
        //   l_bottom_start = (d^h - 1)/(d-1)
        let h = (((d - 1.0) * n + 1.0).log(d)).floor() as u32 - 1;
        let l_bottom_start = (ARITY.pow(h) - 1) / (ARITY - 1);

        for &idx in update_set.iter() {
            if idx.0 >= l_bottom_start {
                bottom.insert(idx);
            } else {
                current.insert(idx);
            }
        }

        // process bottom set first, then merge with top set
        let bottom_parents: FxHashSet<TreeIndex> = bottom
            .into_iter()
            .map(|child_idx| self.parent_index(child_idx).unwrap())
            .collect();
        for p_idx in bottom_parents.iter() {
            f(self, *p_idx);
        }

        current.extend(bottom_parents.into_iter());

        // process until current is empty
        while !current.is_empty() {
            current = current
                .into_iter()
                .filter_map(|child_idx| self.parent_index(child_idx))
                .collect::<FxHashSet<_>>();
            for p_idx in current.iter() {
                f(self, *p_idx);
            }
        }
    }

    pub fn apply_updates_from_single<F: Fn(&mut Self, TreeIndex)>(
        &mut self,
        source: TreeIndex,
        f: F,
    ) {
        let mut maybe_parent = self.parent_index(source);
        while let Some(parent) = maybe_parent {
            f(self, parent);
            maybe_parent = self.parent_index(parent);
        }
    }

    #[inline(always)]
    pub fn one_step_recompute<T>(parent: TreeIndex, tree: &mut [T])
    where
        T: for<'a> std::iter::Sum<&'a T>,
    {
        let start = parent.0 * ARITY + 1;
        let end = (start + ARITY).min(tree.len());
        tree[parent] = tree[start..end].iter().sum();
    }

    #[inline(always)]
    pub fn one_step_recompute_with_timestamp<T, F>(
        parent: TreeIndex,
        tree: &mut [T],
        timestamps: &[usize],
        cur_timestamp: usize,
        mut fallback: F,
    ) where
        T: Copy + std::ops::Add<Output = T>,
        F: FnMut(TreeIndex) -> T,
    {
        let start = parent.0 * ARITY + 1;
        if start >= tree.len() {
            return;
        }
        let end = (start + ARITY).min(tree.len());

        let total = (start..end)
            .map(|idx| {
                if timestamps[idx] == cur_timestamp {
                    tree[idx]
                } else {
                    fallback(TreeIndex(idx))
                }
            })
            .reduce(|a, b| a + b);

        if let Some(total) = total {
            tree[parent] = total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DynamicClustering, Float, TreeIndex, Volume};
    use crate::{alg::cluster, diff::NodeOps};

    type TestClustering = DynamicClustering<2, usize>;

    fn vol(value: f64) -> Volume {
        Volume::new(Float::from(value))
    }

    fn vols(values: &[f64]) -> Vec<Volume> {
        values.iter().copied().map(vol).collect()
    }

    fn fresh_ops(nodes: &[usize], degrees: &[f64]) -> NodeOps<usize> {
        NodeOps {
            created_fresh: (nodes.to_vec(), degrees.to_vec()),
            ..Default::default()
        }
    }

    #[test]
    fn insert_new_nodes_handles_height_changes() {
        let mut clustering = TestClustering::new(Float::from(0.0), 1, 1, 1, cluster);

        clustering.insert_fresh_nodes(&fresh_ops(&[1, 2], &[1.0, 2.0]));
        assert_eq!(clustering.num_leaves(), 2);
        assert_eq!(clustering.num_internal_nodes(), 1);
        assert_eq!(clustering.tree_data.volume[TreeIndex(0)], vol(3.0));
        assert_eq!(clustering.tree_data.size[TreeIndex(0)], 2);

        assert_eq!(
            clustering.node_to_tree_map.get(&1).copied(),
            Some(TreeIndex(1))
        );
        assert_eq!(
            clustering.node_to_tree_map.get(&2).copied(),
            Some(TreeIndex(2))
        );

        clustering.insert_fresh_nodes(&fresh_ops(&[3], &[3.0]));
        assert_eq!(clustering.num_leaves(), 3);
        assert_eq!(clustering.num_internal_nodes(), 2);

        assert_eq!(
            clustering.node_to_tree_map.get(&1).copied(),
            Some(TreeIndex(3))
        );
        assert_eq!(
            clustering.node_to_tree_map.get(&2).copied(),
            Some(TreeIndex(2))
        );
        assert_eq!(
            clustering.node_to_tree_map.get(&3).copied(),
            Some(TreeIndex(4))
        );

        assert_eq!(clustering.tree_data.volume[TreeIndex(1)], vol(4.0));
        assert_eq!(clustering.tree_data.size[TreeIndex(1)], 2);
        assert_eq!(clustering.tree_data.volume[TreeIndex(0)], vol(6.0));
        assert_eq!(clustering.tree_data.size[TreeIndex(0)], 3);
    }

    #[test]
    fn rebuild_from_leaves_updates_multiple_levels() {
        let mut clustering = TestClustering::new(Float::from(0.0), 1, 1, 1, cluster);
        clustering.insert_fresh_nodes(&fresh_ops(
            &[10, 11, 12, 13, 14],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        ));

        for (idx, value) in [(5usize, 10.0), (6, 20.0), (7, 7.0), (8, 9.0)] {
            clustering.tree_data.volume[TreeIndex(idx)] = vol(value);
        }

        clustering.rebuild_from_leaves(5, 9);

        assert_eq!(clustering.tree_data.volume[TreeIndex(2)], vol(30.0));
        assert_eq!(clustering.tree_data.size[TreeIndex(2)], 2);
        assert_eq!(clustering.tree_data.volume[TreeIndex(3)], vol(16.0));
        assert_eq!(clustering.tree_data.size[TreeIndex(3)], 2);
        assert_eq!(clustering.tree_data.volume[TreeIndex(1)], vol(17.0));
        assert_eq!(clustering.tree_data.size[TreeIndex(1)], 3);
        assert_eq!(clustering.tree_data.volume[TreeIndex(0)], vol(47.0));
        assert_eq!(clustering.tree_data.size[TreeIndex(0)], 5);
    }
}
