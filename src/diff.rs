use raphtory::prelude::*;
use raphtory::db::graph::views::deletion_graph::PersistentGraph;

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use raphtory::core::entities::VID;
use raphtory::db::graph::edge::EdgeView;


#[derive(PartialEq, PartialOrd, Debug)]
pub enum EdgeOp{
    Update(VID,VID, f64), // Add/remove this amount (src,dst,weight)
    Delete(VID,VID) // Clear the contribution of this edge (src,dst)
}
#[derive(PartialEq, PartialOrd, Debug, Copy, Clone)]
enum TemporalOp{
    Update(i64, f64), // update to this amount (time, delta)
    Delete(i64) // Clear the contribution of this edge (time)
}




/// For holding update diffs to be applied to an external data structure
/// Applying `snapshot_diffs[0]` to an external data structure
/// moves it from the initial “empty” state to the snapshot at `snapshot_times[0]`.
/// Applying `snapshot_diffs[i]` given the state at `snapshot_times[i-1]`
/// moves you to the state at `snapshot_times[i]`.
#[derive(Debug)]
pub struct SnapshotDiffs{
    pub snapshot_times: Vec<i64>,
    pub snapshot_diffs: Vec<Vec<EdgeOp>>
}

impl SnapshotDiffs{
    pub fn iter(&self) -> impl Iterator<Item = (&i64, &Vec<EdgeOp>)>{
        self.snapshot_times.iter().zip(self.snapshot_diffs.iter())
    }
}




/// For a single edge, return a list of (start, stop, value) intervals for `prop`.
/// `stop` is exclusive.
fn build_timeline(edge: EdgeView<&PersistentGraph>,prop: &str) -> Vec<(i64,i64, f64)>{
    edge.explode().into_iter()
    .filter(|x| x.properties().get(prop).is_some())
    .map(|e|{
        (
            e.earliest_time().unwrap(),
            e.latest_time().unwrap(),
            e.properties().get(prop).unwrap().as_f64().unwrap()
        )
    }).collect()
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
        let cur_val = if idx < timeline.len()
            && snap_t >= timeline[idx].0
            && snap_t < timeline[idx].1
        {
            timeline[idx].2
        } else {
            0.0
        };

        let mut ops = Vec::new();
        if cur_val == prev_val {
            // No change.
        } else if cur_val == 0.0 {
            // Transition from >0 to 0.
            ops.push(TemporalOp::Delete(snap_t));
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
pub fn build_snapshot_diffs(graph: &PersistentGraph, start:i64, end: i64, step_size: usize, prop_name:&str) -> Result<SnapshotDiffs>{

    let graph_start = graph.earliest_time();
    let graph_end = graph.latest_time();



    if graph_start.is_none()|| graph_end.is_none(){
        return Err(anyhow!(
            "No edges in the graph!"
        ));
    }
    let graph_start = graph_start.unwrap();
    let graph_end = graph_end.unwrap();
    if graph_start == graph_end{
        return Err(anyhow!(
            "Snapshots don't make sense for a graph whose edges only live during the instant at {graph_start}"
        ))
    }
    if start<graph_start || end > graph_end{
        return Err(anyhow!(
            "Snapshot range [{start}, {end}) is outside the graph range [{graph_start}, {graph_end})"
        ));
    }

    if step_size == 0{
        return Err(anyhow!(
            "step_size must be greater than zero"
        ))
    }

    let snapshot_times = (start..end).step_by(step_size).collect::<Vec<_>>();

    // for the final reduce when we merge edge snapshot diffs.
    let identity = || snapshot_times.iter().map(|_| Vec::new()).collect::<Vec<Vec<EdgeOp>>>();

    // For each edge, compute per-snapshot temporal ops, then convert them to EdgeOp
    // (delta updates / deletes), then merge the edge snapshots together.
    let snapshot_diffs = graph.edges().iter().par_bridge().map(|edge|{
        let src = edge.edge.src();
        let dst = edge.edge.dst();
        let timeline = build_timeline(edge, prop_name);
        snapshot_ops_for_timeline(&timeline, &snapshot_times).into_iter().map(|buckets|{
            buckets.into_iter().map(|action|{
                match action{
                    TemporalOp::Update(_,x ) => EdgeOp::Update(src, dst, x),
                    TemporalOp::Delete(_) => EdgeOp::Delete(src, dst)
                }
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>()
    }).reduce(identity , |mut acc: Vec<Vec<EdgeOp>>,b: Vec<Vec<EdgeOp>>|{
        for (x,y) in acc.iter_mut().zip(b){
            x.extend(y);
        }
        acc
    });

    Ok(SnapshotDiffs { snapshot_times, snapshot_diffs })
}




// ChatGPT built tests. Most important one is at the end

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use raphtory::db::graph::views::deletion_graph::PersistentGraph;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // --- pure unit tests on snapshot_ops_for_timeline -----------------------

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
            TemporalOp::Delete(_) => {}
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
            .add_edge(
                t,
                &src,
                &dst,
                [("w", Prop::F64(old_edge_weight + w))],
                None,
            )
            .unwrap();
    }

    fn apply_snapshot_ops(
        state: &mut HashMap<(VID, VID), f64>,
        ops: &[EdgeOp],
    ) {
        for op in ops {
            match *op {
                EdgeOp::Update(u, v, delta) => {
                    *state.entry((u, v)).or_insert(0.0) += delta;
                }
                EdgeOp::Delete(u, v) => {
                    state.remove(&(u, v));
                }
            }
        }
    }

    fn snapshot_weights(
        graph: &PersistentGraph,
        t: i64,
    ) -> HashMap<(VID, VID), f64> {
        let mut out = HashMap::new();
        let view = graph.at(t);
        for e in view.edges().iter() {
            let w = e
                .properties()
                .get("w")
                .unwrap()
                .as_f64()
                .unwrap();
            let src: VID = e.edge.src();
            let dst: VID = e.edge.dst();
            out.insert((src, dst), w);
        }
        out
    }

    fn assert_diffs_match_graph(graph: &PersistentGraph, diffs: &SnapshotDiffs) {
        let mut external_state: HashMap<(VID, VID), f64> = HashMap::new();

        for (idx, t) in diffs.snapshot_times.iter().enumerate() {
            apply_snapshot_ops(&mut external_state, &diffs.snapshot_diffs[idx]);
            let expected = snapshot_weights(graph, *t);
            assert_eq!(
                external_state, expected,
                "Mismatch at snapshot time {t}"
            );
        }
    }

    fn assert_full_range_diffs_match_graph(
        graph: &PersistentGraph,
        step_size: usize,
        prop: &str,
    ) {
        let graph_start = graph.earliest_time().unwrap();
        let graph_end = graph.latest_time().unwrap();
        let diffs =
            build_snapshot_diffs(graph, graph_start, graph_end, step_size, prop).unwrap();
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

        let diffs = build_snapshot_diffs(&graph, 1, 6, 10, "w").unwrap();
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
        let diffs = build_snapshot_diffs(&graph, start, end, 5, "w").unwrap();

        assert_diffs_match_graph(&graph, &diffs);
    }

    #[test]
    fn weight_reduced_to_zero_produces_delete_diff() {
        let graph = PersistentGraph::new();

        add_weight(&graph, 0, "u", "v", 3.0);
        graph.add_edge(5, "u", "v", [("w", Prop::F64(0.0))], None).unwrap();

        assert_full_range_diffs_match_graph(&graph, 3, "w");
    }

    #[test]
    fn multiple_edges_changing_in_same_snapshot() {
        let graph = PersistentGraph::new();

        add_weight(&graph, 10, "a", "b", 1.0);
        add_weight(&graph, 10, "b", "c", 2.0);
        add_weight(&graph, 10, "c", "a", 3.0);
        add_weight(&graph, 20, "a", "b", 0.0);

        assert_full_range_diffs_match_graph(&graph, 5, "w");
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
                graph.delete_edge(t, src, dst, None).unwrap();
            }
        }

        assert_full_range_diffs_match_graph(&graph, 3, "w");
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
            TemporalOp::Delete(_) => {}
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
        let identity = || vec![Vec::<EdgeOp>::new(), Vec::<EdgeOp>::new()];

        let timeline_a = vec![(0, 30, 1.0)];
        let ops_a = snapshot_ops_for_timeline(&timeline_a, &snapshot_times);

        let timeline_b = vec![(0, 30, 2.0)];
        let ops_b = snapshot_ops_for_timeline(&timeline_b, &snapshot_times);

        let per_edge_a: Vec<Vec<EdgeOp>> = ops_a
            .into_iter()
            .map(|ops_for_snapshot| {
                ops_for_snapshot
                    .into_iter()
                    .map(|op| match op {
                        TemporalOp::Update(_, delta) => EdgeOp::Update(VID(1), VID(2), delta),
                        TemporalOp::Delete(_) => EdgeOp::Delete(VID(1), VID(2)),
                    })
                    .collect()
            })
            .collect();

        let per_edge_b: Vec<Vec<EdgeOp>> = ops_b
            .into_iter()
            .map(|ops_for_snapshot| {
                ops_for_snapshot
                    .into_iter()
                    .map(|op| match op {
                        TemporalOp::Update(_, delta) => EdgeOp::Update(VID(3), VID(4), delta),
                        TemporalOp::Delete(_) => EdgeOp::Delete(VID(3), VID(4)),
                    })
                    .collect()
            })
            .collect();

        let aggregated = [per_edge_a, per_edge_b]
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

        let per_edge_a: Vec<Vec<EdgeOp>> = ops_a
            .into_iter()
            .map(|ops_for_snapshot| {
                ops_for_snapshot
                    .into_iter()
                    .map(|op| match op {
                        TemporalOp::Update(_, delta) => EdgeOp::Update(VID(1), VID(2), delta),
                        TemporalOp::Delete(_) => EdgeOp::Delete(VID(1), VID(2)),
                    })
                    .collect()
            })
            .collect();

        let per_edge_b: Vec<Vec<EdgeOp>> = ops_b
            .into_iter()
            .map(|ops_for_snapshot| {
                ops_for_snapshot
                    .into_iter()
                    .map(|op| match op {
                        TemporalOp::Update(_, delta) => EdgeOp::Update(VID(3), VID(4), delta),
                        TemporalOp::Delete(_) => EdgeOp::Delete(VID(3), VID(4)),
                    })
                    .collect()
            })
            .collect();

        let identity =
            || vec![Vec::<EdgeOp>::new(), Vec::<EdgeOp>::new(), Vec::<EdgeOp>::new()];

        let aggregated = [per_edge_a, per_edge_b]
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
                EdgeOp::Delete(_, _) => panic!("No deletes expected at snapshot 0"),
            }
        }
        assert!(approx_eq(*map0.get(&(VID(1), VID(2))).unwrap(), 1.0));
        assert!(approx_eq(*map0.get(&(VID(3), VID(4))).unwrap(), 5.0));

        let mut a_delta = 0.0;
        let mut b_deleted = false;
        for op in &aggregated[1] {
            match *op {
                EdgeOp::Update(u, v, delta) if (u, v) == (VID(1), VID(2)) => a_delta += delta,
                EdgeOp::Delete(u, v) if (u, v) == (VID(3), VID(4)) => b_deleted = true,
                _ => {}
            }
        }
        assert!(approx_eq(a_delta, 2.0));
        assert!(b_deleted);

        let mut a_deleted_again = false;
        let mut b_anything = false;
        for op in &aggregated[2] {
            match *op {
                EdgeOp::Delete(u, v) if (u, v) == (VID(1), VID(2)) => a_deleted_again = true,
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
        let res = build_snapshot_diffs(&graph, 0, 10, 1, "w");
        assert!(res.is_err());
    }

    #[test]
    fn build_snapshot_diffs_errors_on_out_of_range_snapshot_window() {
        let graph = PersistentGraph::new();
        add_weight(&graph, 10, "u", "v", 1.0);
        let graph_start = graph.earliest_time().unwrap();
        let graph_end = graph.latest_time().unwrap();

        let res = build_snapshot_diffs(&graph, graph_start - 1, graph_end, 1, "w");
        assert!(res.is_err());

        let res = build_snapshot_diffs(&graph, graph_start, graph_end + 1, 1, "w");
        assert!(res.is_err());
    }

    #[test]
    fn build_snapshot_diffs_errors_on_zero_step() {
        let graph = PersistentGraph::new();
        add_weight(&graph, 10, "u", "v", 1.0);
        let graph_start = graph.earliest_time().unwrap();
        let graph_end = graph.latest_time().unwrap();

        let res = build_snapshot_diffs(&graph, graph_start, graph_end, 0, "w");
        assert!(res.is_err());
    }

    #[test]
    fn build_snapshot_diffs_errors_when_graph_has_zero_duration() {
        let graph = PersistentGraph::new();
        add_weight(&graph, 10, "u", "v", 1.0);

        let start = graph.earliest_time().unwrap();
        let end = graph.latest_time().unwrap();
        assert_eq!(start, end);

        let res = build_snapshot_diffs(&graph, start, end, 1, "w");
        assert!(res.is_err());
    }
}