

mod tests;
mod diff;
mod snapshot_clustering;
mod alg;
use core::time;


use raphtory::{core::entities::VID, prelude::*};
use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use indicatif::ProgressBar;
use std::time::{Instant, Duration};

use tests::{
    generate_commands,
    GeneratedCommands,
    Instruction
};

use diff::build_snapshot_diffs;
use snapshot_clustering::{MyClustering};

use crate::{alg::DynamicClustering, snapshot_clustering::SnapshotClusteringAlg};



fn main(){
    let num_nodes = 1_000_000;
    let num_updates = 100_000_000;
    let GeneratedCommands { nodes, operations } = generate_commands(42424242, num_nodes, num_updates, 0.75, 0.9, 0.5);

    let graph = PersistentGraph::new();

    let pb = ProgressBar::new(num_updates as u64);
    pb.set_style(indicatif::ProgressStyle::default_bar()
        .template("{spinner:.green} {bar:.green/yellow} {decimal_bytes_per_sec} {eta} [{elapsed_precise}] ").unwrap());

    let t0 = Instant::now();

    for command in operations{
        pb.inc(1);
        match command{
            Instruction::Insert(t,u_idx,v_idx,w) => {
                let u = &nodes[u_idx];
                let v = &nodes[v_idx];
                let old_edge_weight = graph.at(t).edge(u,v)
                    .map(|e| e.properties().get("w").unwrap().as_f64().unwrap())
                    .unwrap_or(0.0);

                graph.add_edge(
                    t,
                    u,
                    v,
                    [("w", Prop::F64(old_edge_weight+w))],
                    None
                ).unwrap();
            },
            Instruction::Delete(t,u_idx,v_idx) => {
                let u = &nodes[u_idx];
                let v = &nodes[v_idx];
                graph.delete_edge(
                    t,
                    u,
                    v,
                    None
                ).unwrap();
            },
            Instruction::DeleteHalf(t,u_idx,v_idx,w) => {
                let u = &nodes[u_idx];
                let v = &nodes[v_idx];
                let old_edge_weight = graph.at(t).edge(u,v)
                    .map(|e| e.properties().get("w").unwrap().as_f64().unwrap())
                    .unwrap_or(0.0);
                let new_weight = (old_edge_weight - w).max(0.0);
                if new_weight == 0.0{
                    graph.delete_edge(
                        t,
                        u,
                        v,
                        None
                    ).unwrap();
                } else {
                    graph.add_edge(
                        t,
                        u,
                        v,
                        [("w", Prop::F64(new_weight))],
                        None
                    ).unwrap();
                }
            },
        }
    }
    pb.finish_with_message("processed updates");
    println!("Graph build time: {:?}", t0.elapsed());

    let start = graph.earliest_time().unwrap();
    let end = graph.latest_time().unwrap();


    println!("start: {start}, end: {end}");


    let t1 = Instant::now();
    let diffs = build_snapshot_diffs(&graph, start, end, 100_000, "w", 1e-9).unwrap();
    println!("Diff build time: {:?}", t1.elapsed());
    println!("Number of diffs: {}", diffs.node_diffs.len());

    let t2 = Instant::now();
    let mut cluster_alg: DynamicClustering<8, VID> = alg::DynamicClustering::new(1000.0.into());
    
    let part = cluster_alg.process_node_diffs(&diffs, &graph);
    println!("Clustering time: {:?}", t2.elapsed());

}
