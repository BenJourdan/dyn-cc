

mod tests;
use raphtory::db::api::properties::internal::InternalTemporalPropertyViewOps;
use raphtory::db::api::view::internal::GraphTimeSemanticsOps;
use raphtory::db::graph::edge::EdgeView;
use tests::{
    generate_commands,
    Instruction
};

use raphtory::prelude::*;
use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use indicatif::ProgressBar;

#[derive(PartialEq, PartialOrd)]
pub enum EdgeOp{
    Update(i64,f64), // Add/remove this amount
    Delete(i64) // Clear the contribution of this edge
}

pub enum Update{
    Insert(i64,String, String, f64),
    Delete(i64,String, String),
}


pub fn build_timeline(edge: EdgeView<&PersistentGraph>,prop: &str) -> Vec<EdgeOp>{



    // let updates = edge.history().into_iter().map(|i|{
    //     (i,edge.at(i).properties().get(prop))
    // }).reduce(|acc,e|{

    // })

    let mut update_iter = edge
    .history().into_iter()
    .map(|t| (t,edge.at(t).properties().get(prop).unwrap().as_f64().unwrap())).peekable();
    let mut del_iter = edge.deletions().into_iter().peekable();

    debug_assert!(update_iter.clone().collect::<Vec<_>>().is_sorted());
    debug_assert!(del_iter.clone().collect::<Vec<_>>().is_sorted());


    let mut result = vec!();
    let mut prev_val = 0.0f64;
    
    while update_iter.peek().is_some() || del_iter.peek().is_some(){
        match (update_iter.peek(),  del_iter.peek()){
            (Some(_), None) => {
                let update = update_iter.next().unwrap();
                let diff = 
                result.push(EdgeOp::Update(update.0,update.1));
            },
            (None, Some(_)) =>{

            }

        }
    }

    result
}

// fn build_diff(graph: &PersistentGraph, window_size: usize) -> Vec<Vec<Update>>{
//     // Process the edges in a persistent graph so window diffs can be 
//     // quickly applied by another data structure

//     let start_time = graph.earliest_time().unwrap();
//     let end_time = graph.latest_time().unwrap();

//     let mut diffs: Vec<Vec<Update>> = (start_time..end_time).step_by(window_size).map(|_| vec!()).collect::<Vec<_>>();

//     for 


//     todo!()
// }

fn main(){
    let num_nodes = 10;
    let num_updates = 100_000;
    let commands = generate_commands(42424242, 100, num_updates, 0.8, 1.0, 0.5);

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

    let end = graph.latest_time().unwrap();



    if let Some(edge) = graph.edges().iter().next(){
            println!("{:?}",edge.history());
            for t in edge.history(){
                let x = edge.at(t).properties().get("w").unwrap().as_f64();
                println!("update at {t}:{x:?}");
            }
            for t in edge.deletions(){
                println!("deletion at {t}");
            }
            build_timeline(edge, "w");
    }


}