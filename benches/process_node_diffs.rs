use core::num;
use std::collections::HashMap;
use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use dyn_cc::{alg::DynamicClustering, snapshot_clustering::DiffGraph};
use dyn_cc::diff::build_snapshot_diffs;
use dyn_cc::snapshot_clustering::SnapshotClusteringAlg;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand::seq::SliceRandom;
use raphtory::core::entities::VID;
use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use raphtory::prelude::*;

// Reuse SBM generator from tests.
use dyn_cc::tests::{
    Instruction, adjusted_rand_index, build_graph, generate_sbm_commands, prepare_diff_workload_sbm,
};

fn bench_process_node_diffs(c: &mut Criterion) {
    // Setup runs once; we only measure process_node_diffs itself.

    let num_clusters = 20;
    let coreset_size = 2048;
    let sampling_seeds = num_clusters * 4;

    let n_per_cluster = 512;

    let (nodes, diffs, mut graph, cluster_labels) = prepare_diff_workload_sbm(
        42,
        n_per_cluster,                                      // nodes per cluster
        num_clusters,                                       // clusters
        0.5,                                                // p_internal
        1.0 / (n_per_cluster as f64 * num_clusters as f64), // q_external
        1,                                                  // multiplier for expected edges
        1.0,                                                // lifetime multiplier
        0.1,                                                // step size
    );

    let subset_size = n_per_cluster * num_clusters / 100;

    // sample a uniform subset of nodes to avoid prefix bias
    let mut rng = StdRng::seed_from_u64(42);
    let mut subset_indices: Vec<usize> = (0..nodes.len()).collect();
    subset_indices.shuffle(&mut rng);
    subset_indices.truncate(subset_size.min(nodes.len()));

    let subset = subset_indices
        .iter()
        .map(|&i| graph.node(&nodes[i]).unwrap().node)
        .collect::<Vec<_>>();

    let subset_labels = subset_indices
        .iter()
        .map(|&i| cluster_labels[i])
        .collect::<Vec<_>>();

    assert_eq!(
        adjusted_rand_index(subset_labels.as_slice(), subset_labels.as_slice()),
        1.0
    );

    c.bench_function("process_node_diffs", |b| {
        b.iter(|| {
            let mut alg: DynamicClustering<64, VID> = DynamicClustering::new(
                1000.0.into(),
                coreset_size,
                sampling_seeds,
                num_clusters,
                Arc::new(dyn_cc::alg::cluster),
                String::from("w")
            );
            let mut graph = DiffGraph::with_capacity(n_per_cluster*num_clusters);

            let partitions = alg.process_node_diffs_with_subset(&diffs, &mut graph, subset.as_slice());
            // let partitions = alg.process_node_diffs(&diffs, &graph);
            let aris: Vec<f64> = {
                partitions
                    .iter()
                    .map(|(t, part)| match part {
                        dyn_cc::snapshot_clustering::PartitionOutput::All(_, _) => unreachable!(),
                        dyn_cc::snapshot_clustering::PartitionOutput::Subset(predicted_labels, _) => {
                            assert!(subset_labels.len() == predicted_labels.len());
                            adjusted_rand_index(
                                subset_labels.as_slice(),
                                predicted_labels.as_slice(),
                            )
                        }
                    })
                    .collect()
            };
            println!("ARIs: {:?}", aris);
            black_box(partitions);
        });
    });
}

criterion_group!(benches, bench_process_node_diffs);
criterion_main!(benches);
