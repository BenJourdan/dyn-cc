mod alg;
mod diff;
mod snapshot_clustering;
mod tests;

use crate::snapshot_clustering::DiffGraph;
use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use raphtory::{core::entities::VID, prelude::*};
use core::num;
use rand::{SeedableRng, rngs::StdRng};
use rand::seq::SliceRandom;
use std::time::Instant;

use diff::build_snapshot_diffs;
use snapshot_clustering::MyClustering;
use tests::{GeneratedCommands, Instruction, build_graph, generate_commands, generate_sbm_commands};

use crate::tests::{adjusted_rand_index, prepare_diff_workload_sbm};
use crate::{alg::DynamicClustering, snapshot_clustering::SnapshotClusteringAlg};

fn main() {
    const ARITY: usize = 64;

    let num_clusters = 8;
    let n_per_cluster = 1024;

    let coreset_size = num_clusters*50;
    let sampling_seeds = num_clusters * 4;



    let t0 = Instant::now();

    let subset_size = num_clusters * n_per_cluster/10;

    let (nodes, diffs, graph, cluster_labels) = prepare_diff_workload_sbm(
        42,
        n_per_cluster,                                      // nodes per cluster
        num_clusters,                                       // clusters
        0.5,                                                // p_internal
        1.0 / (2.0*n_per_cluster as f64 * num_clusters as f64), // q_external
        4,                                                  // multiplier for expected edges
        10.0,                                                // lifetime multiplier
        0.05,                                                // step size
    );

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

    println!("Command/Graph/Diff build time: {:?}", t0.elapsed());
    println!("Number of diffs: {}", diffs.node_diffs.len());

    let mut cluster_alg: DynamicClustering<ARITY, VID> = alg::DynamicClustering::new(
        10000.0.into(),
        coreset_size,
        sampling_seeds,
        num_clusters,
        alg::cluster,
    );

    let t2 = Instant::now();


    let mut graph = DiffGraph::with_capacity(n_per_cluster*num_clusters);

    let partitions = cluster_alg.process_node_diffs_with_subset(&diffs, &mut graph, subset.as_slice());
    let aris: Vec<f64> = {
        partitions
            .iter()
            .map(|(_t, part)| match part {
                crate::snapshot_clustering::PartitionOutput::All(_) => unreachable!(),
                crate::snapshot_clustering::PartitionOutput::Subset(predicted_labels) => {
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

    println!("Clustering time: {:?}", t2.elapsed());
}
