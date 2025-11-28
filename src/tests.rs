use indicatif::ProgressBar;
use rand::prelude::*;
use raphtory::core::entities::VID;
use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use raphtory::prelude::*;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};

use crate::diff::build_snapshot_diffs;

pub fn adjusted_rand_index(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    assert_eq!(
        labels_true.len(),
        labels_pred.len(),
        "label arrays must be the same length"
    );
    if labels_true == labels_pred {
        return 1.0;
    }
    let n = labels_true.len();
    if n < 2 {
        return 1.0;
    }

    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut count_true: HashMap<usize, usize> = HashMap::new();
    let mut count_pred: HashMap<usize, usize> = HashMap::new();

    for (&t, &p) in labels_true.iter().zip(labels_pred.iter()) {
        *contingency.entry((t, p)).or_insert(0) += 1;
        *count_true.entry(t).or_insert(0) += 1;
        *count_pred.entry(p).or_insert(0) += 1;
    }

    let choose2 = |x: usize| -> f64 { (x as f64) * ((x as f64) - 1.0) / 2.0 };

    let sum_comb_c = contingency.values().fold(0.0, |acc, &v| acc + choose2(v));
    let sum_comb_true = count_true.values().fold(0.0, |acc, &v| acc + choose2(v));
    let sum_comb_pred = count_pred.values().fold(0.0, |acc, &v| acc + choose2(v));

    let total_pairs = choose2(n);
    let expected_index = (sum_comb_true * sum_comb_pred) / total_pairs;
    let max_index = 0.5 * (sum_comb_true + sum_comb_pred);

    if (max_index - expected_index).abs() < f64::EPSILON {
        0.0
    } else {
        (sum_comb_c - expected_index) / (max_index - expected_index)
    }
}

pub fn build_graph(nodes: &[String], operations: &[Instruction]) -> PersistentGraph {
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
            Instruction::DeletePartial(t, u_idx, v_idx, w) => {
                let u = &nodes[u_idx];
                let v = &nodes[v_idx];
                if w == 0.0 {
                    let _ = graph.delete_edge(t, u, v, None);
                    let _ = graph.delete_edge(t, v, u, None);
                } else {
                    graph
                        .add_edge(t, u, v, [("w", Prop::F64(w))], None)
                        .unwrap();
                    graph
                        .add_edge(t, v, u, [("w", Prop::F64(w))], None)
                        .unwrap();
                }
            }
        }
    }

    graph
}

pub fn prepare_diff_workload_sbm(
    seed: u64,
    n_per_cluster: usize,
    k_clusters: usize,
    p_internal: f64,
    q_external: f64,
    n_multiplier: usize,
    lifetime_multiplier: f64,
    step_size: f64,
) -> (
    Vec<String>,
    crate::diff::SnapshotDiffs<VID>,
    PersistentGraph,
    Vec<usize>,
) {
    let (cmds, _, cluster_labels) = generate_sbm_commands(
        seed,
        n_per_cluster,
        k_clusters,
        p_internal,
        q_external,
        n_multiplier,
        lifetime_multiplier,
    );
    let graph = build_graph(&cmds.nodes, &cmds.operations);

    let start = graph.earliest_time().expect("graph has edges");
    let end = graph.latest_time().expect("graph has edges");

    let step_size = (step_size * (end - start) as f64) as usize;
    println!("num commands: {}", cmds.operations.len());
    println!("timespan: {}", end - start);
    println!("Step size: {}", step_size);

    let diffs = build_snapshot_diffs(&graph, (start + end) / 2, end, step_size, "w", 1e-9)
        .expect("snapshot diffs should build");

    (cmds.nodes, diffs, graph, cluster_labels)
}

pub struct GeneratedCommands {
    pub nodes: Vec<String>,
    pub operations: Vec<Instruction>,
}

pub enum Instruction {
    Insert(i64, usize, usize, f64),
    Delete(i64, usize, usize),
    DeletePartial(i64, usize, usize, f64),
}

pub fn generate_commands(
    seed: u64,
    num_nodes: usize,
    num_updates: usize,
    insert_prob: f64,
    full_delete_prob: f64,
    increment_prob: f64,
) -> GeneratedCommands {
    generate_commands_fast(
        seed,
        num_nodes,
        num_updates,
        insert_prob,
        full_delete_prob,
        increment_prob,
    )
}

pub fn generate_commands_fast(
    seed: u64,
    num_nodes: usize,
    num_updates: usize,
    insert_prob: f64,
    full_delete_prob: f64,
    increment_prob: f64,
) -> GeneratedCommands {
    let mut rng = StdRng::seed_from_u64(seed);

    let nodes: Vec<String> = (0..num_nodes).map(|i| format!("Node{}", i)).collect();

    // Store the edges we inserted/updated (both orientations for symmetry)
    let mut known_edges =
        HashMap::<(usize, usize), f64>::with_capacity((num_updates as f32 * 1.5) as usize);
    // Keep a flat list of keys to sample deletions in O(1) instead of iter().choose()
    let mut edge_keys: Vec<(usize, usize)> = Vec::with_capacity(num_updates * 2);
    let mut edge_index: HashMap<(usize, usize), usize> = HashMap::with_capacity(num_updates * 2);

    let mut add_edge_key =
        |key: (usize, usize),
         edge_keys: &mut Vec<(usize, usize)>,
         edge_index: &mut HashMap<(usize, usize), usize>| {
            let idx = edge_keys.len();
            edge_keys.push(key);
            edge_index.insert(key, idx);
        };

    let mut remove_edge_key =
        |key: (usize, usize),
         edge_keys: &mut Vec<(usize, usize)>,
         edge_index: &mut HashMap<(usize, usize), usize>| {
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

    // Store the operations we are going to perform:
    let mut operations: Vec<Instruction> = Vec::with_capacity(num_updates);

    let pb = ProgressBar::new(num_updates as u64);
    pb.set_style(indicatif::ProgressStyle::default_bar()
        .template("{spinner:.green} {bar:.green/yellow} {decimal_bytes_per_sec} {eta} [{elapsed_precise}] ").unwrap());
    // .progress_chars("##-"));

    let mut counter = 0i64;

    for _ in 0..num_updates {
        pb.inc(1);
        match rng.random_range(0.0..1.0) < insert_prob {
            true => {
                // Insert an edge
                let u = rng.random_range(0..num_nodes);
                let v = rng.random_range(0..num_nodes);
                if u == v {
                    // don't insert self loops
                    continue;
                }
                let w: f64 = rng.random_range(1.0..10.0);
                operations.push(Instruction::Insert(counter, u, v, w));
                // track both directions so deletions can pick either orientation
                for &(a, b) in [(u, v), (v, u)].iter() {
                    match known_edges.entry((a, b)) {
                        Entry::Occupied(mut e) => {
                            *e.get_mut() += w;
                        }
                        Entry::Vacant(e) => {
                            e.insert(w);
                            add_edge_key((a, b), &mut edge_keys, &mut edge_index);
                        }
                    }
                }
            }
            false => {
                // Delete an edge
                if !edge_keys.is_empty() {
                    let idx = rng.random_range(0..edge_keys.len());
                    let (u, v) = edge_keys[idx];
                    let w = *known_edges.get(&(u, v)).unwrap();
                    // with probability 0.5 delete the entire edge
                    if rng.random_range(0.0..1.0) < full_delete_prob {
                        operations.push(Instruction::Delete(counter, u, v));
                        known_edges.remove(&(u, v));
                        known_edges.remove(&(v, u));
                        remove_edge_key((u, v), &mut edge_keys, &mut edge_index);
                        remove_edge_key((v, u), &mut edge_keys, &mut edge_index);
                    } else {
                        // delete half the weight: new_w is the post-delete weight
                        let new_w: f64 = w / 2.0;
                        operations.push(Instruction::DeletePartial(counter, u, v, new_w));
                        // update the known edges to the new weight
                        known_edges.insert((u, v), new_w);
                        known_edges.insert((v, u), new_w);
                    }
                }
            }
        }
        if rng.random_range(0.0..1.0) < increment_prob {
            counter += 1;
        }
    }
    pb.finish_with_message("Done generating instructions");
    GeneratedCommands { nodes, operations }
}

/// Generate an SBM-style stream of edge updates with finite lifetimes.
pub fn generate_sbm_commands(
    seed: u64,
    n_per_cluster: usize,
    k_clusters: usize,
    p_internal: f64,
    q_external: f64,
    n_multiplier: usize,
    lifetime_multiplier: f64,
) -> (GeneratedCommands, usize, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let total_nodes = n_per_cluster * k_clusters;
    let nodes: Vec<String> = (0..k_clusters)
        .flat_map(|c| (0..n_per_cluster).map(move |i| format!("C{}_{}", c, i)))
        .collect();

    // node indices grouped by cluster for fast sampling
    let cluster_nodes: Vec<Vec<usize>> = (0..k_clusters)
        .map(|c| {
            let start = c * n_per_cluster;
            (start..start + n_per_cluster).collect()
        })
        .collect();

    let mut edge_weights: HashMap<(usize, usize), f64> = HashMap::new();
    // Track additions to expire after `lifetime` steps.
    let mut expirations: VecDeque<(i64, (usize, usize))> = VecDeque::new();

    // Expected edge count of an SBM(k, n_per_cluster, p_internal, q_external).
    let expected_internal = (k_clusters as f64)
        * (n_per_cluster as f64)
        * ((n_per_cluster - 1) as f64)
        * 0.5
        * p_internal;
    let expected_external = (k_clusters * (k_clusters - 1)) as f64
        * 0.5
        * (n_per_cluster * n_per_cluster) as f64
        * q_external;
    let expected_edges = expected_internal + expected_external;

    let num_updates = ((n_multiplier as f64) * expected_edges).ceil() as usize;
    let lifetime_steps = (lifetime_multiplier * expected_edges).ceil() as i64;

    let mut operations: Vec<Instruction> = Vec::with_capacity(num_updates * 2);
    let mut t: i64 = 0;

    let total_weight = expected_internal + expected_external;
    let internal_prob = if total_weight > 0.0 {
        expected_internal / total_weight
    } else {
        0.5
    };

    let mut pick_internal = |rng: &mut StdRng, cluster_nodes: &[Vec<usize>]| -> (usize, usize) {
        let c = rng.random_range(0..cluster_nodes.len());
        let cluster = &cluster_nodes[c];
        if cluster.len() == 1 {
            return (cluster[0], cluster[0]);
        }
        let a = rng.random_range(0..cluster.len());
        let mut b = rng.random_range(0..cluster.len());
        while b == a {
            b = rng.random_range(0..cluster.len());
        }
        (cluster[a], cluster[b])
    };

    let mut pick_cross = |rng: &mut StdRng, cluster_nodes: &[Vec<usize>]| -> (usize, usize) {
        if cluster_nodes.len() < 2 {
            return pick_internal(rng, cluster_nodes);
        }
        let c1 = rng.random_range(0..cluster_nodes.len());
        let mut c2 = rng.random_range(0..cluster_nodes.len());
        while c2 == c1 {
            c2 = rng.random_range(0..cluster_nodes.len());
        }
        let u_cluster = &cluster_nodes[c1];
        let v_cluster = &cluster_nodes[c2];
        (
            u_cluster[rng.random_range(0..u_cluster.len())],
            v_cluster[rng.random_range(0..v_cluster.len())],
        )
    };

    for _ in 0..num_updates {
        // expire old edges
        while let Some(&(added_at, (u, v))) = expirations.front() {
            if t - added_at >= lifetime_steps {
                expirations.pop_front();
                let key = if u <= v { (u, v) } else { (v, u) };
                if let Some(w) = edge_weights.get_mut(&key) {
                    let new_w = (*w - 1.0).max(0.0);
                    *w = new_w;
                    if new_w == 0.0 {
                        edge_weights.remove(&key);
                        operations.push(Instruction::Delete(t, u, v));
                    } else {
                        operations.push(Instruction::DeletePartial(t, u, v, new_w));
                    }
                }
            } else {
                break;
            }
        }

        // choose edge type and endpoints
        let (u, v) = if rng.random_range(0.0..1.0) < internal_prob {
            pick_internal(&mut rng, &cluster_nodes)
        } else {
            pick_cross(&mut rng, &cluster_nodes)
        };
        if u == v {
            continue;
        }

        // add/update edge
        operations.push(Instruction::Insert(t, u, v, 1.0));
        let key = if u <= v { (u, v) } else { (v, u) };
        *edge_weights.entry(key).or_insert(0.0) += 1.0;
        expirations.push_back((t, (u, v)));

        // Always advance time.
        t += 1;
    }

    let mut cluster_labels = Vec::with_capacity(total_nodes);
    for c in 0..k_clusters {
        cluster_labels.extend(std::iter::repeat(c).take(n_per_cluster));
    }

    (
        GeneratedCommands { nodes, operations },
        expected_edges as usize,
        cluster_labels,
    )
}

#[test]
fn diffs_replay_matches_graph_snapshots() {
    // Build a random-ish graph, derive diffs, replay only node ops into a simple model,
    // and compare node presence/degrees against the Raphtory graph at each snapshot.
    use crate::diff::{NodeOps, build_snapshot_diffs};
    use rustc_hash::FxHashMap as HashMap;

    let num_nodes = 1000;
    let num_updates = 10_000;
    let cmds = generate_commands_fast(424242, num_nodes, num_updates, 0.7, 0.4, 0.25);

    let graph = PersistentGraph::new();
    // Apply generated operations to the Raphtory graph.
    for cmd in &cmds.operations {
        match *cmd {
            Instruction::Insert(t, u_idx, v_idx, w) => {
                let u = &cmds.nodes[u_idx];
                let v = &cmds.nodes[v_idx];
                let old_w = graph
                    .at(t)
                    .edge(u, v)
                    .map(|e| e.properties().get("w").unwrap().as_f64().unwrap())
                    .unwrap_or(0.0);
                graph
                    .add_edge(t, u, v, [("w", Prop::F64(old_w + w))], None)
                    .unwrap();
                graph
                    .add_edge(t, v, u, [("w", Prop::F64(old_w + w))], None)
                    .unwrap();
            }
            Instruction::Delete(t, u_idx, v_idx) => {
                let u = &cmds.nodes[u_idx];
                let v = &cmds.nodes[v_idx];
                let _ = graph.delete_edge(t, u, v, None);
                let _ = graph.delete_edge(t, v, u, None);
            }
            Instruction::DeletePartial(t, u_idx, v_idx, w) => {
                let u = &cmds.nodes[u_idx];
                let v = &cmds.nodes[v_idx];
                if w == 0.0 {
                    let _ = graph.delete_edge(t, u, v, None);
                    let _ = graph.delete_edge(t, v, u, None);
                } else {
                    graph
                        .add_edge(t, u, v, [("w", Prop::F64(w))], None)
                        .unwrap();
                    graph
                        .add_edge(t, v, u, [("w", Prop::F64(w))], None)
                        .unwrap();
                }
            }
        }
    }

    let start = graph.earliest_time().unwrap();
    let end = graph.latest_time().unwrap();
    let step = ((end - start) / 20).max(1); // keep test time reasonable
    let diffs = build_snapshot_diffs(&graph, start, end, step as usize, "w", 1e-9).unwrap();

    #[derive(Default)]
    struct Model {
        nodes: HashMap<VID, f64>,
    }

    impl Model {
        fn apply_node_ops(&mut self, ops: &NodeOps<VID>) {
            for (&n, &deg) in ops.created_fresh.0.iter().zip(ops.created_fresh.1.iter()) {
                self.nodes.insert(n, deg);
            }
            for (&n, &deg) in ops.created_stale.0.iter().zip(ops.created_stale.1.iter()) {
                self.nodes.insert(n, deg);
            }
            for (&n, &deg) in ops.modified.0.iter().zip(ops.modified.1.iter()) {
                self.nodes.insert(n, deg);
            }
            for &n in &ops.deleted.0 {
                self.nodes.remove(&n);
            }
        }
    }

    fn snapshot_state(graph: &PersistentGraph, time: i64) -> HashMap<VID, f64> {
        let mut nodes = HashMap::default();
        let mut seen = std::collections::HashSet::new();
        let view = graph.at(time);

        // Seed node map so isolated nodes appear with degree 0.
        for n in view.nodes().iter() {
            nodes.insert(n.node, 0.0);
        }

        // Treat edges as undirected: only count each unordered pair once, but
        // add its weight to both endpoints.
        for e in view.edges().iter() {
            let src = e.src().node;
            let dst = e.dst().node;
            let key = if src <= dst { (src, dst) } else { (dst, src) };
            if !seen.insert(key) {
                continue;
            }
            let w = e
                .properties()
                .get("w")
                .and_then(|p| p.as_f64())
                .unwrap_or(0.0);
            *nodes.entry(src).or_insert(0.0) += w;
            *nodes.entry(dst).or_insert(0.0) += w;
        }

        nodes
    }

    let mut model: Model = Model::default();
    for (t_node, node_ops) in diffs.iter_node_diffs() {
        model.apply_node_ops(node_ops);

        let g_nodes = snapshot_state(&graph, *t_node);
        assert_eq!(
            model.nodes.len(),
            g_nodes.len(),
            "node set size mismatch at snapshot {}",
            t_node
        );
        for (v, gdeg) in g_nodes.iter() {
            let mdeg = model.nodes.get(v).copied().unwrap_or(-1.0);
            assert!(
                (gdeg - mdeg).abs() < 1e-9,
                "node degree mismatch for {:?} at snapshot {}: graph {}, model {}",
                v,
                t_node,
                gdeg,
                mdeg
            );
        }
    }
}

#[test]
fn dynamic_clustering_degrees_track_graph() {
    use crate::alg::DynamicClustering;
    use crate::diff::{NodeOps, build_snapshot_diffs};
    use crate::snapshot_clustering::{GraphLike, SnapshotClusteringAlg};
    use rustc_hash::FxHashMap as HashMap;

    let num_nodes = 128;
    let num_updates = 5_000;
    let cmds = generate_commands_fast(737373, num_nodes, num_updates, 0.7, 0.4, 0.25);

    let graph = PersistentGraph::new();
    // Apply generated operations to the Raphtory graph.
    for cmd in &cmds.operations {
        match *cmd {
            Instruction::Insert(t, u_idx, v_idx, w) => {
                let u = &cmds.nodes[u_idx];
                let v = &cmds.nodes[v_idx];
                let old_w = graph
                    .at(t)
                    .edge(u, v)
                    .map(|e| e.properties().get("w").unwrap().as_f64().unwrap())
                    .unwrap_or(0.0);
                graph
                    .add_edge(t, u, v, [("w", Prop::F64(old_w + w))], None)
                    .unwrap();
                graph
                    .add_edge(t, v, u, [("w", Prop::F64(old_w + w))], None)
                    .unwrap();
            }
            Instruction::Delete(t, u_idx, v_idx) => {
                let u = &cmds.nodes[u_idx];
                let v = &cmds.nodes[v_idx];
                let _ = graph.delete_edge(t, u, v, None);
                let _ = graph.delete_edge(t, v, u, None);
            }
            Instruction::DeletePartial(t, u_idx, v_idx, w) => {
                let u = &cmds.nodes[u_idx];
                let v = &cmds.nodes[v_idx];
                if w == 0.0 {
                    let _ = graph.delete_edge(t, u, v, None);
                    let _ = graph.delete_edge(t, v, u, None);
                } else {
                    graph
                        .add_edge(t, u, v, [("w", Prop::F64(w))], None)
                        .unwrap();
                    graph
                        .add_edge(t, v, u, [("w", Prop::F64(w))], None)
                        .unwrap();
                }
            }
        }
    }

    let start = graph.earliest_time().unwrap();
    let end = graph.latest_time().unwrap();
    let step = ((end - start) / 20).max(1); // keep test time reasonable
    let diffs = build_snapshot_diffs(&graph, start, end, step as usize, "w", 1e-9).unwrap();

    fn graph_degree_map(graph: &PersistentGraph, time: i64) -> HashMap<VID, f64> {
        let mut nodes = HashMap::default();
        let mut seen = std::collections::HashSet::new();
        let view = graph.at(time);
        for n in view.nodes().iter() {
            nodes.insert(n.node, 0.0);
        }
        for e in view.edges().iter() {
            let src = e.src().node;
            let dst = e.dst().node;
            let key = if src <= dst { (src, dst) } else { (dst, src) };
            if !seen.insert(key) {
                continue;
            }
            let w = e
                .properties()
                .get("w")
                .and_then(|p| p.as_f64())
                .unwrap_or(0.0);
            *nodes.entry(src).or_insert(0.0) += w;
            *nodes.entry(dst).or_insert(0.0) += w;
        }
        nodes
    }

    // Initialize clustering with arbitrary sigma/coreset params (not used for degree tracking).
    let mut clustering: DynamicClustering<8, VID> =
        DynamicClustering::new(1.0.into(), 16, 4, 2, crate::alg::cluster);

    for (t_node, node_ops) in diffs.iter_node_diffs() {
        clustering.apply_node_ops(*t_node, node_ops, &graph);

        let g_nodes = graph_degree_map(&graph, *t_node);

        // Build model degrees from clustering state: any leaf with nonzero volume.
        let mut model = HashMap::default();
        for (node, idx) in clustering.node_to_tree_map.iter() {
            let deg = clustering.tree_data.volume[idx.0].0;
            if deg != 0.0 {
                model.insert(*node, deg);
            }
        }

        assert_eq!(
            model.len(),
            g_nodes.len(),
            "node set size mismatch at snapshot {}",
            t_node
        );

        for (v, gdeg) in g_nodes.iter() {
            let mdeg = model.get(v).copied().unwrap_or((-1.0).into());
            assert!(
                (gdeg - mdeg.0).abs() < 1e-9,
                "node degree mismatch for {:?} at snapshot {}: graph {}, model {}",
                v,
                t_node,
                gdeg,
                mdeg
            );
        }
    }
}
