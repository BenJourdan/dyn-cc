use raphtory::prelude::*;
use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use rand::prelude::*;
use std::collections::HashMap;
use indicatif::ProgressBar;

pub enum Instruction{
    Insert(i64,String, String, f64),
    Delete(i64,String, String),
    DeleteHalf(i64,String, String, f64),
}

pub fn generate_commands(seed: u64, num_nodes: usize, num_updates: usize, insert_prob: f64, full_delete_prob: f64,increment_prob: f64) -> Vec<Instruction>{

    let mut rng = StdRng::seed_from_u64(seed);

    let nodes: Vec<String> = (0..num_nodes)
    .map(|i| format!("Node{}", i))
    .collect();

    // Store the edges we inserted/updated
    let mut known_edges = HashMap::<(String,String), f64>::with_capacity((num_updates as f32*1.5) as usize);

    // Store the operations we are going to perform:
    let mut operations: Vec<Instruction> = Vec::with_capacity(num_updates);

    let pb = ProgressBar::new(num_updates as u64);
    pb.set_style(indicatif::ProgressStyle::default_bar()
        .template("{spinner:.green} {bar:.green/yellow} {decimal_bytes_per_sec} {eta} [{elapsed_precise}] ").unwrap());
        // .progress_chars("##-"));
    
    let mut counter = 0i64;

    for _ in 0..num_updates {
        pb.inc(1);
        match rng.random_range(0.0..1.0) < insert_prob{
            true => {
                // Insert an edge
                let u = nodes[rng.random_range(0..num_nodes)].clone();
                let v = nodes[rng.random_range(0..num_nodes)].clone();
                if u == v{
                    // don't insert self loops
                    continue;
                }
                let w: f64 = rng.random_range(1.0..10.0);
                operations.push(Instruction::Insert(counter, u.clone(), v.clone(), w));
                known_edges.entry((u.clone(),v.clone())).and_modify(|e| *e += w).or_insert(w);
                known_edges.entry((v.clone(),u.clone())).and_modify(|e| *e += w).or_insert(w);

            },
            false => {
                // Delete an edge
                if !known_edges.is_empty(){

                    let entry = known_edges.iter().choose(&mut rng).unwrap();
                    let u = entry.0.0.clone();
                    let v = entry.0.1.clone();
                    let w = entry.1.clone();
                    // with probability 0.5 delete the entire edge
                    if rng.random_range(0.0..1.0) < full_delete_prob{
                        operations.push(Instruction::Delete(counter,u.clone(), v.clone()));
                        known_edges.remove(&(u.clone(),v.clone()));
                        known_edges.remove(&(v.clone(),u.clone()));
                    }else{
                        // delete half the weight
                        let new_w: f64 = w/2.0;
                        operations.push(Instruction::DeleteHalf(counter,u.clone(), v.clone(), new_w));
                        // update the known edges
                        known_edges.entry((u.clone(),v.clone())).and_modify(|e| *e -= new_w).or_insert(new_w);
                        known_edges.entry((v.clone(),u.clone())).and_modify(|e| *e -= new_w).or_insert(new_w);
                    }
                }
            }
        }
        if rng.random_range(0.0..1.0) < increment_prob{
            counter += 1;
        }
    }
    pb.finish_with_message("Done generating instructions");
    operations
}




#[test]
fn test_graph(){

}



