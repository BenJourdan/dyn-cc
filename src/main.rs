

mod tests;
mod diff;
mod snapshot_clustering;
use core::time;


use raphtory::prelude::*;
use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use indicatif::ProgressBar;

use tests::{
    generate_commands,
    Instruction
};

use diff::build_snapshot_diffs;
use snapshot_clustering::{MyClustering};

use crate::snapshot_clustering::SnapshotClusteringAlg;



fn main(){
    let num_nodes = 1_000;
    let num_updates = 100_000;
    let commands = generate_commands(42424242, num_nodes, num_updates, 0.99, 1.0, 0.5);

    let graph = PersistentGraph::new();

    let pb = ProgressBar::new(num_updates as u64);
    pb.set_style(indicatif::ProgressStyle::default_bar()
        .template("{spinner:.green} {bar:.green/yellow} {decimal_bytes_per_sec} {eta} [{elapsed_precise}] ").unwrap());

    for command in commands{
        pb.inc(1);
        match command{
            Instruction::Insert(t,u,v,w) => {
                let old_edge_weight = graph.at(t).edge(&u,&v)
                    .map(|e| e.properties().get("w").unwrap().as_f64().unwrap())
                    .unwrap_or(0.0);

                graph.add_edge(
                    t,
                    &u,
                    &v,
                    [("w", Prop::F64(old_edge_weight+w))],
                    None
                ).unwrap();
            },
            Instruction::Delete(t,u,v) => {
                graph.delete_edge(
                    t,
                    &u,
                    &v,
                    None
                ).unwrap();
            },
            Instruction::DeleteHalf(t,u,v,w) => {
                let old_edge_weight = graph.at(t).edge(&u,&v)
                    .map(|e| e.properties().get("w").unwrap().as_f64().unwrap())
                    .unwrap_or(0.0);
                let new_weight = (old_edge_weight - w).max(0.0);
                if new_weight == 0.0{
                    graph.delete_edge(
                        t,
                        &u,
                        &v,
                        None
                    ).unwrap();
                } else {
                    graph.add_edge(
                        t,
                        &u,
                        &v,
                        [("w", Prop::F64(new_weight))],
                        None
                    ).unwrap();
                }
            },
        }
    }
    pb.finish_with_message("processed updates");

    let start = graph.earliest_time().unwrap();
    let end = graph.latest_time().unwrap();


    println!("{end}");

    let diffs = build_snapshot_diffs(&graph, start, end, 500, "w").unwrap();

    // for (i,diff) in diffs.iter().take(10){
    //     println!("{i:?}: {:?}", diff.iter().take(10).collect::<Vec<_>>());
    // }

    let mut cluster_alg = MyClustering::new();

    cluster_alg.process_diffs_with(&diffs, |time,partition|{
        println!("processed time: {time}, with a partition of size {}",partition.len());
    });

}