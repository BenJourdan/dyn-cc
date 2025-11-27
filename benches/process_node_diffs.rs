use std::collections::HashMap;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use dyn_cc::alg::DynamicClustering;
use dyn_cc::diff::build_snapshot_diffs;
use dyn_cc::snapshot_clustering::SnapshotClusteringAlg;
use rand::{Rng, SeedableRng, rngs::StdRng};
use raphtory::core::entities::VID;
use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use raphtory::prelude::*;

#[derive(Clone, Copy)]
enum Instruction {
    Insert(i64, usize, usize, f64),
    Delete(i64, usize, usize),
}

fn generate_workload(
    seed: u64,
    num_nodes: usize,
    num_updates: usize,
    insert_prob: f64,
) -> (Vec<String>, Vec<Instruction>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let nodes: Vec<String> = (0..num_nodes).map(|i| format!("Node{i}")).collect();
    let mut operations = Vec::with_capacity(num_updates);

    // Keep a lightweight edge set so deletes only target existing edges.
    let mut edge_keys: Vec<(usize, usize)> = Vec::new();
    let mut edge_index: HashMap<(usize, usize), usize> = HashMap::new();

    let mut add_edge_key = |edge_keys: &mut Vec<(usize, usize)>,
                            edge_index: &mut HashMap<(usize, usize), usize>,
                            key: (usize, usize)| {
        if edge_index.contains_key(&key) {
            return;
        }
        let idx = edge_keys.len();
        edge_keys.push(key);
        edge_index.insert(key, idx);
    };

    let mut remove_edge_key = |edge_keys: &mut Vec<(usize, usize)>,
                               edge_index: &mut HashMap<(usize, usize), usize>,
                               key: (usize, usize)| {
        if let Some(idx) = edge_index.remove(&key) {
            let last = edge_keys.len() - 1;
            edge_keys.swap(idx, last);
            if idx != last {
                let moved = edge_keys[idx];
                edge_index.insert(moved, idx);
            }
            edge_keys.pop();
        }
    };

    for step in 0..num_updates {
        let time = step as i64;
        let do_insert = edge_keys.is_empty() || rng.random_bool(insert_prob);

        if do_insert {
            let u = rng.random_range(0..num_nodes);
            let mut v = rng.random_range(0..num_nodes);
            if u == v {
                v = (v + 1) % num_nodes;
            }
            let w = rng.random_range(0.5..3.0);
            operations.push(Instruction::Insert(time, u, v, w));

            let key = if u <= v { (u, v) } else { (v, u) };
            add_edge_key(&mut edge_keys, &mut edge_index, key);
        } else {
            let idx = rng.random_range(0..edge_keys.len());
            let (u, v) = edge_keys[idx];
            operations.push(Instruction::Delete(time, u, v));
            remove_edge_key(&mut edge_keys, &mut edge_index, (u, v));
        }
    }

    (nodes, operations)
}

fn build_graph(nodes: &[String], operations: &[Instruction]) -> PersistentGraph {
    let graph = PersistentGraph::new();

    for op in operations {
        match *op {
            Instruction::Insert(t, u_idx, v_idx, w) => {
                let u = &nodes[u_idx];
                let v = &nodes[v_idx];
                let old_edge_weight = graph
                    .at(t)
                    .edge(u, v)
                    .map(|e| e.properties().get("w").unwrap().as_f64().unwrap())
                    .unwrap_or(0.0);

                graph
                    .add_edge(t, u, v, [("w", Prop::F64(old_edge_weight + w))], None)
                    .unwrap();
                graph
                    .add_edge(t, v, u, [("w", Prop::F64(old_edge_weight + w))], None)
                    .unwrap();
            }
            Instruction::Delete(t, u_idx, v_idx) => {
                let u = &nodes[u_idx];
                let v = &nodes[v_idx];
                // Ignore missing edges; this only happens if a delete races a duplicate insert.
                let _ = graph.delete_edge(t, u, v, None);
                let _ = graph.delete_edge(t, v, u, None);
            }
        }
    }

    graph
}

fn prepare_diff_workload(
    seed: u64,
    num_nodes: usize,
    num_updates: usize,
    step_size: usize,
) -> (dyn_cc::diff::SnapshotDiffs<VID>, PersistentGraph) {
    let (nodes, operations) = generate_workload(seed, num_nodes, num_updates, 0.65);
    let graph = build_graph(&nodes, &operations);

    let start = graph.earliest_time().expect("graph has edges");
    let end = graph.latest_time().expect("graph has edges");

    let diffs = build_snapshot_diffs(&graph, (start + end) / 2, end, step_size, "w", 1e-9)
        .expect("snapshot diffs should build");

    (diffs, graph)
}

fn bench_process_node_diffs(c: &mut Criterion) {
    // Setup runs once; we only measure process_node_diffs itself.
    let (diffs, graph) = prepare_diff_workload(42, 20_000, 1_000_000, 100_000);

    c.bench_function("process_node_diffs", |b| {
        b.iter(|| {
            let mut alg: DynamicClustering<64, VID> =
                DynamicClustering::new(1000.0.into(), 1024, 64, 8, dyn_cc::alg::cluster);
            let partitions = alg.process_node_diffs(&diffs, &graph);
            black_box(partitions);
        });
    });
}

criterion_group!(benches, bench_process_node_diffs);
criterion_main!(benches);
