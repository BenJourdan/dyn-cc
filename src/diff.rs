use std::collections::{HashMap, HashSet};

use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use raphtory::prelude::*;

use anyhow::{Result, anyhow};
use raphtory::core::entities::VID;
use raphtory::db::graph::edge::EdgeView;
use rayon::prelude::*;

// used to tell external data structures if just the edge has been deleted, or if the node is now a singleton
// The value is the weight of the edge prior to deletion
#[derive(PartialEq, PartialOrd, Debug)]
pub enum DeletionEvent {
    Edge(f64),
    Node(f64),
}

#[derive(PartialEq, PartialOrd, Debug)]
pub enum EdgeOp<V> {
    Update(V, V, f64),     // Add/remove this amount (src,dst,weight)
    DeleteEdge(V, V, f64), // Clear the contribution of this edge (src,dst, edge_weight)
}

impl<V: Copy> EdgeOp<V> {
    pub fn src(&self) -> V {
        match self {
            EdgeOp::Update(s, _, _) => *s,
            EdgeOp::DeleteEdge(s, _, _) => *s,
        }
    }
    pub fn dst(&self) -> V {
        match self {
            EdgeOp::Update(_, d, _) => *d,
            EdgeOp::DeleteEdge(_, d, _) => *d,
        }
    }
}

/// Extended edge operations that take into account whether the source and/or destination nodes are
/// created/deleted as part of the edge operation. This can be used by algorithms which would otherwise
/// have to work this out themselves (e.g., by maintaining a node degree map).
#[derive(PartialEq, PartialOrd, Debug)]
pub enum ExtendedEdgeOp<V> {
    SrcDstPresentUpdate(V, V, f64), // Both endpoints exist. Add/remove this amount to the edge(src,dst,weight)
    SrcMissingUpdate(V, V, f64), // Src node is missing. Add/remove this amount to the edge (src,dst,weight)
    DstMissingUpdate(V, V, f64), // Dst node is missing. Add/remove this amount to the edge (src,dst,weight)
    BothMissingUpdate(V, V, f64), // Both endpoints are missing. Add/remove this amount to the edge (src,dst,weight)

    SrcDstPresentDelete(V, V, f64), // Both endpoints exist. Clear the contribution of this edge (src,dst, edge_weight)
    SrcRemoveDelete(V, V, f64), // Src is dangling after this edge deletion. Remove edge and src node (src,dst, edge_weight)
    DstRemoveDelete(V, V, f64), // Dst is dangling after this edge deletion. Remove edge and dst node (src,dst, edge_weight)
    BothRemoveDelete(V, V, f64), // Both endpoints are dangling after this edge deletion. Remove edge and both nodes (src,dst, edge_weight)
}

#[derive(PartialEq, PartialOrd, Debug)]
pub struct NodeOps<V> {
    pub created_fresh: (Vec<V>, Vec<f64>),
    pub created_stale: (Vec<V>, Vec<f64>),
    pub modified: (Vec<V>, Vec<f64>),
    pub deleted: (Vec<V>, Vec<f64>),
}

impl<V> Default for NodeOps<V> {
    fn default() -> Self {
        Self {
            created_fresh: (Vec::new(), Vec::new()),
            created_stale: (Vec::new(), Vec::new()),
            modified: (Vec::new(), Vec::new()),
            deleted: (Vec::new(), Vec::new()),
        }
    }
}

impl<V: Copy> ExtendedEdgeOp<V> {
    pub fn edge_op(&self) -> EdgeOp<V> {
        match self {
            ExtendedEdgeOp::SrcDstPresentUpdate(s, d, w)
            | ExtendedEdgeOp::SrcMissingUpdate(s, d, w)
            | ExtendedEdgeOp::DstMissingUpdate(s, d, w)
            | ExtendedEdgeOp::BothMissingUpdate(s, d, w) => EdgeOp::Update(*s, *d, *w),

            ExtendedEdgeOp::SrcDstPresentDelete(s, d, x)
            | ExtendedEdgeOp::SrcRemoveDelete(s, d, x)
            | ExtendedEdgeOp::DstRemoveDelete(s, d, x)
            | ExtendedEdgeOp::BothRemoveDelete(s, d, x) => EdgeOp::DeleteEdge(*s, *d, *x),
        }
    }
}

fn near_zero(value: f64, epsilon: f64) -> bool {
    value.abs() < epsilon
}

pub fn extend_diffs<V: Copy + std::hash::Hash + Eq>(
    diffs: &Vec<Vec<EdgeOp<V>>>,
    epsilon: f64,
) -> (Vec<Vec<ExtendedEdgeOp<V>>>, Vec<NodeOps<V>>) {
    fn edge_weight<V: Copy + std::hash::Hash + Eq>(
        graph: &HashMap<V, HashMap<V, f64>>,
        src: V,
        dst: V,
    ) -> f64 {
        graph
            .get(&src)
            .and_then(|neighbors| neighbors.get(&dst))
            .copied()
            .unwrap_or(0.0)
    }

    fn set_edge_weight<V: Copy + std::hash::Hash + Eq>(
        graph: &mut HashMap<V, HashMap<V, f64>>,
        src: V,
        dst: V,
        weight: f64,
        epsilon: f64,
    ) {
        if near_zero(weight, epsilon) {
            if let Some(neighbors) = graph.get_mut(&src) {
                neighbors.remove(&dst);
                if neighbors.is_empty() {
                    graph.remove(&src);
                }
            }
        } else {
            graph.entry(src).or_default().insert(dst, weight);
        }
    }

    fn node_degree<V: Copy + std::hash::Hash + Eq>(
        graph: &HashMap<V, HashMap<V, f64>>,
        node: V,
    ) -> f64 {
        graph
            .get(&node)
            .map(|neighbors| neighbors.values().sum())
            .unwrap_or(0.0)
    }

    let mut graph: HashMap<V, HashMap<V, f64>> = HashMap::new();
    let mut ever_seen: HashSet<V> = HashSet::new();

    let mut extended_diffs = Vec::with_capacity(diffs.len());
    let mut node_diffs = Vec::with_capacity(diffs.len());

    for diff in diffs {
        let mut extended_diff = Vec::with_capacity(diff.len());
        let mut pre_snapshot_degrees: HashMap<V, f64> = HashMap::new();
        let mut seen_nodes: HashSet<V> = HashSet::new();
        let mut touched_nodes: Vec<V> = Vec::new();

        let mut record_node = |node: V, graph: &HashMap<V, HashMap<V, f64>>| {
            if seen_nodes.insert(node) {
                pre_snapshot_degrees.insert(node, node_degree(graph, node));
                touched_nodes.push(node);
            }
        };

        for op in diff {
            match op {
                EdgeOp::Update(s, d, w) => {
                    record_node(*s, &graph);
                    record_node(*d, &graph);

                    let src_present = !near_zero(node_degree(&graph, *s), epsilon);
                    let dst_present = !near_zero(node_degree(&graph, *d), epsilon);

                    match (src_present, dst_present) {
                        (false, false) => {
                            extended_diff.push(ExtendedEdgeOp::BothMissingUpdate(*s, *d, *w))
                        }
                        (false, true) => {
                            extended_diff.push(ExtendedEdgeOp::SrcMissingUpdate(*s, *d, *w))
                        }
                        (true, false) => {
                            extended_diff.push(ExtendedEdgeOp::DstMissingUpdate(*s, *d, *w))
                        }
                        (true, true) => {
                            extended_diff.push(ExtendedEdgeOp::SrcDstPresentUpdate(*s, *d, *w))
                        }
                    };

                    let current_weight = edge_weight(&graph, *s, *d);
                    let mut new_weight = current_weight + *w;
                    if near_zero(new_weight, epsilon) {
                        new_weight = 0.0;
                    }
                    set_edge_weight(&mut graph, *s, *d, new_weight, epsilon);
                    set_edge_weight(&mut graph, *d, *s, new_weight, epsilon);
                }
                EdgeOp::DeleteEdge(s, d, x) => {
                    record_node(*s, &graph);
                    record_node(*d, &graph);

                    // Remove only the contribution of this directed edge; keep any opposite-direction weight.
                    let mut new_weight = edge_weight(&graph, *s, *d) - *x;
                    if near_zero(new_weight, epsilon) {
                        new_weight = 0.0;
                    }
                    set_edge_weight(&mut graph, *s, *d, new_weight, epsilon);
                    set_edge_weight(&mut graph, *d, *s, new_weight, epsilon);

                    let src_deleted = near_zero(node_degree(&graph, *s), epsilon);
                    let dst_deleted = near_zero(node_degree(&graph, *d), epsilon);

                    match (src_deleted, dst_deleted) {
                        (false, false) => {
                            extended_diff.push(ExtendedEdgeOp::SrcDstPresentDelete(*s, *d, *x))
                        }
                        (true, false) => {
                            extended_diff.push(ExtendedEdgeOp::SrcRemoveDelete(*s, *d, *x));
                        }
                        (false, true) => {
                            extended_diff.push(ExtendedEdgeOp::DstRemoveDelete(*s, *d, *x));
                        }
                        (true, true) => {
                            extended_diff.push(ExtendedEdgeOp::BothRemoveDelete(*s, *d, *x));
                        }
                    };
                }
            }
        }

        let mut node_ops = NodeOps::default();
        for node in touched_nodes {
            let before = *pre_snapshot_degrees.get(&node).unwrap_or(&0.0);
            let after = node_degree(&graph, node);

            let before_present = !near_zero(before, epsilon);
            let after_present = !near_zero(after, epsilon);

            if before_present && !after_present {
                node_ops.deleted.0.push(node);
                node_ops.deleted.1.push(before);
            } else if !before_present && after_present {
                if ever_seen.insert(node) {
                    node_ops.created_fresh.0.push(node);
                    node_ops.created_fresh.1.push(after);
                } else {
                    node_ops.created_stale.0.push(node);
                    node_ops.created_stale.1.push(after);
                }
            } else if before_present && after_present && (before - after).abs() > epsilon {
                node_ops.modified.0.push(node);
                node_ops.modified.1.push(after);
            }
        }

        node_diffs.push(node_ops);
        extended_diffs.push(extended_diff);
    }
    (extended_diffs, node_diffs)
}

#[derive(PartialEq, PartialOrd, Debug, Copy, Clone)]
enum TemporalOp {
    Update(i64, f64), // update to this amount (time, delta)
    Delete(i64, f64), // Clear the contribution of this edge (time, edge_weight)
}

/// For holding update diffs to be applied to an external data structure
/// Applying `snapshot_diffs[0]` to an external data structure
/// moves it from the initial “empty” state to the snapshot at `snapshot_times[0]`.
/// Applying `snapshot_diffs[i]` given the state at `snapshot_times[i-1]`
/// moves you to the state at `snapshot_times[i]`.
#[derive(Debug)]
pub struct SnapshotDiffs<V> {
    pub snapshot_times: Vec<i64>,
    pub snapshot_diffs: Vec<Vec<ExtendedEdgeOp<V>>>,
    pub node_diffs: Vec<NodeOps<V>>,
}

impl<V> SnapshotDiffs<V> {
    pub fn iter_edge_diffs(&self) -> impl Iterator<Item = (&i64, &Vec<ExtendedEdgeOp<V>>)> {
        self.snapshot_times.iter().zip(self.snapshot_diffs.iter())
    }
    pub fn iter_node_diffs(&self) -> impl Iterator<Item = (&i64, &NodeOps<V>)> {
        self.snapshot_times.iter().zip(self.node_diffs.iter())
    }
}

/// For a single edge, return a list of (start, stop, value) intervals for `prop`.
/// `stop` is exclusive.
fn build_timeline(edge: EdgeView<&PersistentGraph>, prop: &str) -> Vec<(i64, i64, f64)> {
    let mut timeline = edge
        .explode()
        .into_iter()
        .filter(|x| x.properties().get(prop).is_some())
        .map(|e| {
            (
                e.earliest_time().unwrap(),
                e.latest_time().unwrap(), // end is exclusive for snapshots
                e.properties().get(prop).unwrap().as_f64().unwrap(),
            )
        })
        .collect::<Vec<_>>();

    timeline.sort_by_key(|(start, _, _)| *start);
    timeline
}

/// Given an edge timeline and a list of snapshot times, return, for each snapshot,
/// a list of `TemporalOp` describing the **delta** needed to move from the previous
/// snapshot to this snapshot for this edge.
///
/// Semantics:
/// - Let `v_j` be the value of the edge at snapshot time `snapshot_times[j]`
///   (0 if absent).
/// - Let `v_{-1} = 0`.
/// - If `v_j == v_{j-1}`: no op at this snapshot.
/// - Else If `v_j > 0`:
///   emit `Update(snapshot_times[j], v_j - v_{j-1})`.
/// - Otherwise `v_j == 0` and `v_{j-1} > 0`:
///   emit `Delete(snapshot_times[j])`.
fn snapshot_ops_for_timeline(
    timeline: &[(i64, i64, f64)],
    snapshot_times: &[i64],
) -> Vec<Vec<TemporalOp>> {
    if snapshot_times.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(snapshot_times.len());

    // Pointer into timeline intervals.
    let mut idx = 0usize;
    let mut prev_val = 0.0_f64;

    for &snap_t in snapshot_times.iter() {
        // Advance intervals that end before or at snap_t.
        while idx < timeline.len() && snap_t >= timeline[idx].1 {
            idx += 1;
        }

        // Determine current value at snap_t.
        let cur_val =
            if idx < timeline.len() && snap_t >= timeline[idx].0 && snap_t < timeline[idx].1 {
                timeline[idx].2
            } else {
                0.0
            };

        let mut ops = Vec::new();
        if cur_val == prev_val {
            // No change.
        } else if cur_val == 0.0 {
            // Transition from >0 to 0.
            // We delete
            ops.push(TemporalOp::Delete(snap_t, prev_val));
        } else {
            // Non-zero value: emit delta.
            let delta = cur_val - prev_val;
            ops.push(TemporalOp::Update(snap_t, delta));
        }

        prev_val = cur_val;
        result.push(ops);
    }

    result
}

/// Build snapshot diffs for the entire graph.
///
/// `start` and `end` define the time range; `step_size` is the spacing between snapshots.
/// We create snapshot times as:
///   snapshot_times = [start, start + step_size, start + 2*step_size, ... , < end]
///
/// `prop_name` is the edge property used as the weight.
pub fn build_snapshot_diffs(
    graph: &PersistentGraph,
    start: i64,
    end: i64,
    step_size: usize,
    prop_name: &str,
    epsilon: f64,
) -> Result<SnapshotDiffs<VID>> {
    let graph_start = graph.earliest_time();
    let graph_end = graph.latest_time();

    if graph_start.is_none() || graph_end.is_none() {
        return Err(anyhow!("No edges in the graph!"));
    }
    let graph_start = graph_start.unwrap();
    let graph_end = graph_end.unwrap();
    if graph_start == graph_end {
        return Err(anyhow!(
            "Snapshots don't make sense for a graph whose edges only live during the instant at {graph_start}"
        ));
    }
    if start < graph_start || end > graph_end {
        return Err(anyhow!(
            "Snapshot range [{start}, {end}) is outside the graph range [{graph_start}, {graph_end})"
        ));
    }

    if step_size == 0 {
        return Err(anyhow!("step_size must be greater than zero"));
    }

    let snapshot_times = (start..end).step_by(step_size).collect::<Vec<_>>();

    // for the final reduce when we merge edge snapshot diffs.
    let identity = || {
        snapshot_times
            .iter()
            .map(|_| Vec::new())
            .collect::<Vec<Vec<EdgeOp<VID>>>>()
    };

    // Build per-edge timelines once and reduce them into snapshot diffs.
    let edge_snapshot_diffs = graph
        .edges()
        .iter()
        .par_bridge()
        .filter_map(|edge| {
            let src: VID = edge.edge.src();
            let dst: VID = edge.edge.dst();

            // Process each undirected pair once.
            if src > dst {
                return None;
            }

            let timeline = build_timeline(edge, prop_name);
            if timeline.is_empty() {
                return None;
            }

            let per_snapshot = snapshot_ops_for_timeline(&timeline, &snapshot_times);
            Some(
                per_snapshot
                    .into_iter()
                    .map(|ops_for_snapshot| {
                        ops_for_snapshot
                            .into_iter()
                            .map(|op| match op {
                                TemporalOp::Update(_, delta) => EdgeOp::Update(src, dst, delta),
                                TemporalOp::Delete(_, x) => EdgeOp::DeleteEdge(src, dst, x),
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .reduce(
            || identity(),
            |mut acc, edge_contrib| {
                for (acc_slot, edge_slot) in acc.iter_mut().zip(edge_contrib.into_iter()) {
                    acc_slot.extend(edge_slot);
                }
                acc
            },
        );

    let (snapshot_diffs, node_diffs) = extend_diffs(&edge_snapshot_diffs, epsilon);

    Ok(SnapshotDiffs {
        snapshot_times,
        snapshot_diffs,
        node_diffs,
    })
}

//MARK: Tests

// ChatGPT built tests. Most important one is at the end

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use raphtory::db::graph::views::deletion_graph::PersistentGraph;
    use std::collections::HashMap;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // --- pure unit tests on snapshot_ops_for_timeline -----------------------

    #[test]
    fn extend_diffs_tracks_node_creation_and_deletion_with_tolerance() {
        let v1 = VID(1);
        let v2 = VID(2);
        let epsilon = 1e-9;
        let tiny = epsilon / 2.0;

        let diffs = vec![
            vec![EdgeOp::Update(v1, v2, 1.0)],
            vec![EdgeOp::Update(v1, v2, -1.0 + tiny)],
            vec![EdgeOp::Update(v1, v2, 0.2)],
            vec![EdgeOp::DeleteEdge(v1, v2, 0.2)],
        ];

        let (extended, node_diffs) = extend_diffs(&diffs, epsilon);
        assert_eq!(extended.len(), diffs.len());
        assert_eq!(node_diffs.len(), diffs.len());

        match extended[0][0] {
            ExtendedEdgeOp::BothMissingUpdate(s, d, w) => {
                assert_eq!((s, d), (v1, v2));
                assert!(approx_eq(w, 1.0));
            }
            _ => panic!("Expected BothMissingUpdate for initial creation"),
        }

        match extended[1][0] {
            ExtendedEdgeOp::SrcDstPresentUpdate(s, d, w) => {
                assert_eq!((s, d), (v1, v2));
                assert!(approx_eq(w, -1.0 + tiny));
            }
            _ => panic!("Expected SrcDstPresentUpdate when nodes already exist"),
        }

        match extended[2][0] {
            ExtendedEdgeOp::BothMissingUpdate(s, d, w) => {
                assert_eq!((s, d), (v1, v2));
                assert!(approx_eq(w, 0.2));
            }
            _ => panic!("Expected BothMissingUpdate after previous near-zero removal"),
        }

        match extended[3][0] {
            ExtendedEdgeOp::BothRemoveDelete(s, d, w) => {
                assert_eq!((s, d), (v1, v2));
                assert!(approx_eq(w, 0.2));
            }
            _ => panic!("Expected BothRemoveDelete when the only edge is deleted"),
        }

        let node_ops_to_map = |ops: &NodeOps<VID>| -> HashMap<VID, (&str, f64)> {
            let mut out = HashMap::new();
            for (v, deg) in ops.created_fresh.0.iter().zip(ops.created_fresh.1.iter()) {
                out.insert(*v, ("created_fresh", *deg));
            }
            for (v, deg) in ops.created_stale.0.iter().zip(ops.created_stale.1.iter()) {
                out.insert(*v, ("created_stale", *deg));
            }
            for (v, deg) in ops.modified.0.iter().zip(ops.modified.1.iter()) {
                out.insert(*v, ("modified", *deg));
            }
            for (v, deg) in ops.deleted.0.iter().zip(ops.deleted.1.iter()) {
                out.insert(*v, ("deleted", *deg));
            }
            out
        };

        let created0 = node_ops_to_map(&node_diffs[0]);
        assert!(matches!(created0.get(&v1), Some(("created_fresh", deg)) if approx_eq(*deg,1.0)));
        assert!(matches!(created0.get(&v2), Some(("created_fresh", deg)) if approx_eq(*deg,1.0)));

        let deleted1 = node_ops_to_map(&node_diffs[1]);
        assert!(matches!(deleted1.get(&v1), Some(("deleted", _))));
        assert!(matches!(deleted1.get(&v2), Some(("deleted", _))));

        let created2 = node_ops_to_map(&node_diffs[2]);
        assert!(matches!(created2.get(&v1), Some(("created_stale", deg)) if approx_eq(*deg,0.2)));
        assert!(matches!(created2.get(&v2), Some(("created_stale", deg)) if approx_eq(*deg,0.2)));

        let deleted3 = node_ops_to_map(&node_diffs[3]);
        assert!(matches!(deleted3.get(&v1), Some(("deleted", deg)) if approx_eq(*deg,0.2)));
        assert!(matches!(deleted3.get(&v2), Some(("deleted", deg)) if approx_eq(*deg,0.2)));
    }

    #[test]
    fn snapshot_ops_constant_timeline() {
        let timeline = vec![(0, 30, 1.0)];
        let snapshot_times = vec![10, 20];

        let per_snapshot = snapshot_ops_for_timeline(&timeline, &snapshot_times);

        assert_eq!(per_snapshot.len(), 2);

        assert_eq!(per_snapshot[0].len(), 1);
        match per_snapshot[0][0] {
            TemporalOp::Update(_, delta) => assert!(approx_eq(delta, 1.0)),
            _ => panic!("Expected Update for first snapshot"),
        }

        assert!(per_snapshot[1].is_empty());
    }

    #[test]
    fn snapshot_ops_update_without_gap() {
        let timeline = vec![(0, 10, 1.0), (10, 20, 3.0)];
        let snapshot_times = vec![5, 15];

        let per_snapshot = snapshot_ops_for_timeline(&timeline, &snapshot_times);

        assert_eq!(per_snapshot.len(), 2);

        assert_eq!(per_snapshot[0].len(), 1);
        match per_snapshot[0][0] {
            TemporalOp::Update(_, delta) => assert!(approx_eq(delta, 1.0)),
            _ => panic!("Expected Update at first snapshot"),
        }

        assert_eq!(per_snapshot[1].len(), 1);
        match per_snapshot[1][0] {
            TemporalOp::Update(_, delta) => assert!(approx_eq(delta, 2.0)),
            _ => panic!("Expected Update at second snapshot"),
        }
    }

    #[test]
    fn snapshot_ops_gap_with_delete_and_readd() {
        let timeline = vec![(0, 10, 1.0), (20, 30, 2.0)];
        let snapshot_times = vec![5, 15, 25];

        let per_snapshot = snapshot_ops_for_timeline(&timeline, &snapshot_times);

        assert_eq!(per_snapshot.len(), 3);

        assert_eq!(per_snapshot[0].len(), 1);
        match per_snapshot[0][0] {
            TemporalOp::Update(_, delta) => assert!(approx_eq(delta, 1.0)),
            _ => panic!("Expected Update at first snapshot"),
        }

        assert_eq!(per_snapshot[1].len(), 1);
        match per_snapshot[1][0] {
            TemporalOp::Delete(_, x) => {}
            _ => panic!("Expected Delete at second snapshot"),
        }

        assert_eq!(per_snapshot[2].len(), 1);
        match per_snapshot[2][0] {
            TemporalOp::Update(_, delta) => assert!(approx_eq(delta, 2.0)),
            _ => panic!("Expected Update at third snapshot"),
        }
    }

    // --- shared helpers for integration-style tests ------------------------

    fn add_weight(graph: &PersistentGraph, t: i64, src: &str, dst: &str, w: f64) {
        let old_edge_weight = graph
            .at(t)
            .edge(&src, &dst)
            .map(|e| e.properties().get("w").unwrap().as_f64().unwrap())
            .unwrap_or(0.0);

        graph
            .add_edge(t, &src, &dst, [("w", Prop::F64(old_edge_weight + w))], None)
            .unwrap();
        graph
            .add_edge(t, &dst, &src, [("w", Prop::F64(old_edge_weight + w))], None)
            .unwrap();
    }

    fn canonical_pair<V: Ord + Copy>(a: V, b: V) -> (V, V) {
        if a <= b { (a, b) } else { (b, a) }
    }

    fn apply_snapshot_ops<V: Copy + Eq + std::hash::Hash + Ord>(
        state: &mut HashMap<(V, V), f64>,
        ops: &[ExtendedEdgeOp<V>],
    ) {
        for op in ops {
            match op.edge_op() {
                EdgeOp::Update(u, v, delta) => {
                    let key = canonical_pair(u, v);
                    *state.entry(key).or_insert(0.0) += delta;
                }
                EdgeOp::DeleteEdge(u, v, x) => {
                    let key = canonical_pair(u, v);
                    state.remove(&key);
                }
            }
        }
    }

    fn snapshot_weights(graph: &PersistentGraph, t: i64) -> HashMap<(VID, VID), f64> {
        let mut out = HashMap::new();
        let view = graph.at(t);
        for e in view.edges().iter() {
            let w = e.properties().get("w").unwrap().as_f64().unwrap();
            let src: VID = e.edge.src();
            let dst: VID = e.edge.dst();
            let key = canonical_pair(src, dst);
            out.insert(key, w);
        }
        out
    }

    fn assert_diffs_match_graph(graph: &PersistentGraph, diffs: &SnapshotDiffs<VID>) {
        let mut external_state: HashMap<(VID, VID), f64> = HashMap::new();

        for (idx, t) in diffs.snapshot_times.iter().enumerate() {
            apply_snapshot_ops(&mut external_state, &diffs.snapshot_diffs[idx]);
            let expected = snapshot_weights(graph, *t);
            if external_state != expected {
                let extra: Vec<_> = external_state
                    .iter()
                    .filter(|(k, v)| expected.get(k) != Some(*v))
                    .collect();
                let missing: Vec<_> = expected
                    .iter()
                    .filter(|(k, v)| external_state.get(k) != Some(*v))
                    .collect();
                if let Some((k, v)) = missing.first() {
                    let canon = *k;
                    let seen_in_ops: usize = diffs.snapshot_diffs[..=idx]
                        .iter()
                        .flat_map(|ops| ops.iter())
                        .filter(|op| {
                            let (u, v) = match op.edge_op() {
                                EdgeOp::Update(u, v, _) | EdgeOp::DeleteEdge(u, v, _) => {
                                    canonical_pair(u, v)
                                }
                            };
                            (u, v) == *canon
                        })
                        .count();
                    panic!(
                        "Mismatch at snapshot time {t}; extra: {}, missing: {} (first missing: {:?} -> {}, occurrences in diffs <= idx: {})",
                        extra.len(),
                        missing.len(),
                        k,
                        v,
                        seen_in_ops
                    );
                } else {
                    panic!(
                        "Mismatch at snapshot time {t}; extra: {}, missing: {}",
                        extra.len(),
                        missing.len()
                    );
                }
            }
        }
    }

    fn assert_full_range_diffs_match_graph(
        graph: &PersistentGraph,
        step_size: usize,
        prop: &str,
        epsilon: f64,
    ) {
        let graph_start = graph.earliest_time().unwrap();
        let graph_end = graph.latest_time().unwrap();
        let diffs =
            build_snapshot_diffs(graph, graph_start, graph_end, step_size, prop, epsilon).unwrap();
        assert!(!diffs.snapshot_times.is_empty());
        assert_diffs_match_graph(graph, &diffs);
    }

    // ============================================================
    // 1. Integration / behaviour tests
    // ============================================================

    #[test]
    fn snapshot_diffs_match_persistent_graph_for_simple_edge() {
        let graph = PersistentGraph::new();

        let src = "u";
        let dst = "v";

        add_weight(&graph, 1, src, dst, 1.0);
        add_weight(&graph, 6, src, dst, 2.0);

        let diffs = build_snapshot_diffs(&graph, 1, 6, 10, "w", EPS).unwrap();
        assert_eq!(diffs.snapshot_times, vec![1]);

        assert_diffs_match_graph(&graph, &diffs);
    }

    #[test]
    fn edge_appears_after_initial_snapshots() {
        let graph = PersistentGraph::new();

        add_weight(&graph, 10, "u", "v", 5.0);
        add_weight(&graph, 20, "u", "v", 10.0);

        let start = graph.earliest_time().unwrap();
        let end = graph.latest_time().unwrap();
        let diffs = build_snapshot_diffs(&graph, start, end, 5, "w", EPS).unwrap();

        assert_diffs_match_graph(&graph, &diffs);
    }

    #[test]
    fn weight_reduced_to_zero_produces_delete_diff() {
        let graph = PersistentGraph::new();

        add_weight(&graph, 0, "u", "v", 3.0);
        graph
            .add_edge(5, "u", "v", [("w", Prop::F64(0.0))], None)
            .unwrap();
        graph
            .add_edge(5, "v", "u", [("w", Prop::F64(0.0))], None)
            .unwrap();

        assert_full_range_diffs_match_graph(&graph, 3, "w", EPS);
    }

    #[test]
    fn multiple_edges_changing_in_same_snapshot() {
        let graph = PersistentGraph::new();

        add_weight(&graph, 10, "a", "b", 1.0);
        add_weight(&graph, 10, "b", "c", 2.0);
        add_weight(&graph, 10, "c", "a", 3.0);
        add_weight(&graph, 20, "a", "b", 0.0);

        assert_full_range_diffs_match_graph(&graph, 5, "w", EPS);
    }

    /// Random-ish graph, deterministic seed: diffs must replay to match graph.at()
    #[test]
    fn pseudo_random_diffs_match_graph_snapshots() {
        let graph = PersistentGraph::new();

        let mut rng = StdRng::seed_from_u64(42);

        const N_VERTS: usize = 100;
        let verts: Vec<String> = (0..N_VERTS).map(|i| format!("v{i}")).collect();

        let t_start = 0_i64;
        let t_end = 1000_i64;

        add_weight(&graph, t_start, &verts[0], &verts[1], 1.0);

        for t in (t_start + 1)..t_end {
            if rng.random_bool(0.5) {
                continue;
            }

            let i = rng.random_range(0..N_VERTS);
            let mut j = rng.random_range(0..N_VERTS);
            if j == i {
                j = (j + 1) % N_VERTS;
            }
            let src = &verts[i];
            let dst = &verts[j];

            if rng.random_bool(0.6) {
                add_weight(&graph, t, src, dst, 1.0);
            } else {
                let _ = graph.delete_edge(t, src, dst, None);
                let _ = graph.delete_edge(t, dst, src, None);
            }
        }

        assert_full_range_diffs_match_graph(&graph, 3, "w", EPS);
    }

    // ============================================================
    // 2. Aggregation tests on pure timelines
    // ============================================================

    #[test]
    fn complex_timeline_diffs_with_gap_and_readd() {
        let timeline = vec![(0, 10, 1.0), (10, 20, 3.0), (30, 40, 4.0)];
        let snapshot_times = vec![5, 15, 25, 35];

        let per_snapshot = snapshot_ops_for_timeline(&timeline, &snapshot_times);

        assert_eq!(per_snapshot.len(), 4);

        assert_eq!(per_snapshot[0].len(), 1);
        match per_snapshot[0][0] {
            TemporalOp::Update(_, delta) => assert!(approx_eq(delta, 1.0)),
            _ => panic!("Expected Update at snapshot 0"),
        }

        assert_eq!(per_snapshot[1].len(), 1);
        match per_snapshot[1][0] {
            TemporalOp::Update(_, delta) => assert!(approx_eq(delta, 2.0)),
            _ => panic!("Expected Update at snapshot 1"),
        }

        assert_eq!(per_snapshot[2].len(), 1);
        match per_snapshot[2][0] {
            TemporalOp::Delete(_, _) => {}
            _ => panic!("Expected Delete at snapshot 2"),
        }

        assert_eq!(per_snapshot[3].len(), 1);
        match per_snapshot[3][0] {
            TemporalOp::Update(_, delta) => assert!(approx_eq(delta, 4.0)),
            _ => panic!("Expected Update at snapshot 3"),
        }
    }

    #[test]
    fn multiple_events_in_same_snapshot_keep_only_last() {
        let timeline = vec![(0, 5, 1.0), (5, 10, 3.0)];
        let snapshot_times = vec![8];

        let per_snapshot = snapshot_ops_for_timeline(&timeline, &snapshot_times);

        assert_eq!(per_snapshot.len(), 1);
        assert_eq!(per_snapshot[0].len(), 1);

        match per_snapshot[0][0] {
            TemporalOp::Update(_, delta) => assert!(approx_eq(delta, 3.0)),
            _ => panic!("Expected single Update at snapshot 0"),
        }
    }

    #[test]
    fn aggregated_diffs_for_two_constant_edges() {
        let snapshot_times = vec![10, 20];
        let identity = || vec![Vec::<EdgeOp<VID>>::new(), Vec::<EdgeOp<VID>>::new()];

        let timeline_a = vec![(0, 30, 1.0)];
        let ops_a = snapshot_ops_for_timeline(&timeline_a, &snapshot_times);

        let timeline_b = vec![(0, 30, 2.0)];
        let ops_b = snapshot_ops_for_timeline(&timeline_b, &snapshot_times);

        let per_edge_a: Vec<Vec<EdgeOp<VID>>> = ops_a
            .into_iter()
            .map(|ops_for_snapshot| {
                ops_for_snapshot
                    .into_iter()
                    .map(|op| match op {
                        TemporalOp::Update(_, delta) => EdgeOp::Update(VID(1), VID(2), delta),
                        TemporalOp::Delete(_, x) => EdgeOp::DeleteEdge(VID(1), VID(2), x),
                    })
                    .collect()
            })
            .collect();

        let per_edge_b: Vec<Vec<EdgeOp<VID>>> = ops_b
            .into_iter()
            .map(|ops_for_snapshot| {
                ops_for_snapshot
                    .into_iter()
                    .map(|op| match op {
                        TemporalOp::Update(_, delta) => EdgeOp::Update(VID(3), VID(4), delta),
                        TemporalOp::Delete(_, x) => EdgeOp::DeleteEdge(VID(3), VID(4), x),
                    })
                    .collect()
            })
            .collect();

        let aggregated =
            [per_edge_a, per_edge_b]
                .into_iter()
                .fold(identity(), |mut acc, edge_contrib| {
                    for (acc_slot, edge_slot) in acc.iter_mut().zip(edge_contrib) {
                        acc_slot.extend(edge_slot);
                    }
                    acc
                });

        assert_eq!(aggregated[0].len(), 2);
        let mut map0: HashMap<(VID, VID), f64> = HashMap::new();
        for op in &aggregated[0] {
            match *op {
                EdgeOp::Update(u, v, delta) => {
                    *map0.entry((u, v)).or_insert(0.0) += delta;
                }
                _ => panic!("No deletes expected at first snapshot"),
            }
        }

        assert!(approx_eq(*map0.get(&(VID(1), VID(2))).unwrap(), 1.0));
        assert!(approx_eq(*map0.get(&(VID(3), VID(4))).unwrap(), 2.0));

        assert!(aggregated[1].is_empty());
    }

    #[test]
    fn aggregated_diffs_for_two_edges_with_updates_and_deletes() {
        let snapshot_times = vec![5, 15, 25];

        let timeline_a = vec![(0, 10, 1.0), (10, 20, 3.0)];
        let ops_a = snapshot_ops_for_timeline(&timeline_a, &snapshot_times);

        let timeline_b = vec![(5, 15, 5.0)];
        let ops_b = snapshot_ops_for_timeline(&timeline_b, &snapshot_times);

        let per_edge_a: Vec<Vec<EdgeOp<VID>>> = ops_a
            .into_iter()
            .map(|ops_for_snapshot| {
                ops_for_snapshot
                    .into_iter()
                    .map(|op| match op {
                        TemporalOp::Update(_, delta) => EdgeOp::Update(VID(1), VID(2), delta),
                        TemporalOp::Delete(_, x) => EdgeOp::DeleteEdge(VID(1), VID(2), x),
                    })
                    .collect()
            })
            .collect();

        let per_edge_b: Vec<Vec<EdgeOp<VID>>> = ops_b
            .into_iter()
            .map(|ops_for_snapshot| {
                ops_for_snapshot
                    .into_iter()
                    .map(|op| match op {
                        TemporalOp::Update(_, delta) => EdgeOp::Update(VID(3), VID(4), delta),
                        TemporalOp::Delete(_, x) => EdgeOp::DeleteEdge(VID(3), VID(4), x),
                    })
                    .collect()
            })
            .collect();

        let identity = || {
            vec![
                Vec::<EdgeOp<VID>>::new(),
                Vec::<EdgeOp<VID>>::new(),
                Vec::<EdgeOp<VID>>::new(),
            ]
        };

        let aggregated =
            [per_edge_a, per_edge_b]
                .into_iter()
                .fold(identity(), |mut acc, edge_contrib| {
                    for (acc_slot, edge_slot) in acc.iter_mut().zip(edge_contrib) {
                        acc_slot.extend(edge_slot);
                    }
                    acc
                });

        let mut map0: HashMap<(VID, VID), f64> = HashMap::new();
        for op in &aggregated[0] {
            match *op {
                EdgeOp::Update(u, v, delta) => {
                    *map0.entry((u, v)).or_insert(0.0) += delta;
                }
                EdgeOp::DeleteEdge(_, _, _) => panic!("No deletes expected at snapshot 0"),
            }
        }
        assert!(approx_eq(*map0.get(&(VID(1), VID(2))).unwrap(), 1.0));
        assert!(approx_eq(*map0.get(&(VID(3), VID(4))).unwrap(), 5.0));

        let mut a_delta = 0.0;
        let mut b_deleted = false;
        for op in &aggregated[1] {
            match *op {
                EdgeOp::Update(u, v, delta) if (u, v) == (VID(1), VID(2)) => a_delta += delta,
                EdgeOp::DeleteEdge(u, v, _) if (u, v) == (VID(3), VID(4)) => b_deleted = true,
                _ => {}
            }
        }
        assert!(approx_eq(a_delta, 2.0));
        assert!(b_deleted);

        let mut a_deleted_again = false;
        let mut b_anything = false;
        for op in &aggregated[2] {
            match *op {
                EdgeOp::DeleteEdge(u, v, _) if (u, v) == (VID(1), VID(2)) => a_deleted_again = true,
                EdgeOp::Update(_, _, _) => b_anything = true,
                _ => {}
            }
        }
        assert!(a_deleted_again);
        assert!(!b_anything);
    }

    // ============================================================
    // 3. Error-path tests
    // ============================================================

    #[test]
    fn build_snapshot_diffs_errors_on_empty_graph() {
        let graph = PersistentGraph::new();
        let res = build_snapshot_diffs(&graph, 0, 10, 1, "w", EPS);
        assert!(res.is_err());
    }

    #[test]
    fn build_snapshot_diffs_errors_on_out_of_range_snapshot_window() {
        let graph = PersistentGraph::new();
        add_weight(&graph, 10, "u", "v", 1.0);
        let graph_start = graph.earliest_time().unwrap();
        let graph_end = graph.latest_time().unwrap();

        let res = build_snapshot_diffs(&graph, graph_start - 1, graph_end, 1, "w", EPS);
        assert!(res.is_err());

        let res = build_snapshot_diffs(&graph, graph_start, graph_end + 1, 1, "w", EPS);
        assert!(res.is_err());
    }

    #[test]
    fn build_snapshot_diffs_errors_on_zero_step() {
        let graph = PersistentGraph::new();
        add_weight(&graph, 10, "u", "v", 1.0);
        let graph_start = graph.earliest_time().unwrap();
        let graph_end = graph.latest_time().unwrap();

        let res = build_snapshot_diffs(&graph, graph_start, graph_end, 0, "w", EPS);
        assert!(res.is_err());
    }

    #[test]
    fn build_snapshot_diffs_errors_when_graph_has_zero_duration() {
        let graph = PersistentGraph::new();
        add_weight(&graph, 10, "u", "v", 1.0);

        let start = graph.earliest_time().unwrap();
        let end = graph.latest_time().unwrap();
        assert_eq!(start, end);

        let res = build_snapshot_diffs(&graph, start, end, 1, "w", EPS);
        assert!(res.is_err());
    }
}
