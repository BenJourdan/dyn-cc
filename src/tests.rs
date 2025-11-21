use raphtory::prelude::*;
use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use rand::prelude::*;
use std::collections::HashMap;
use indicatif::ProgressBar;
use std::collections::hash_map::Entry;

pub struct GeneratedCommands {
    pub nodes: Vec<String>,
    pub operations: Vec<Instruction>,
}

pub enum Instruction{
    Insert(i64,usize, usize, f64),
    Delete(i64,usize, usize),
    DeleteHalf(i64,usize, usize, f64),
}

pub fn generate_commands(seed: u64, num_nodes: usize, num_updates: usize, insert_prob: f64, full_delete_prob: f64,increment_prob: f64) -> GeneratedCommands{
    generate_commands_fast(seed, num_nodes, num_updates, insert_prob, full_delete_prob, increment_prob)
}

pub fn generate_commands_fast(seed: u64, num_nodes: usize, num_updates: usize, insert_prob: f64, full_delete_prob: f64,increment_prob: f64) -> GeneratedCommands{

    let mut rng = StdRng::seed_from_u64(seed);

    let nodes: Vec<String> = (0..num_nodes)
    .map(|i| format!("Node{}", i))
    .collect();

    // Store the edges we inserted/updated (both orientations for symmetry)
    let mut known_edges = HashMap::<(usize,usize), f64>::with_capacity((num_updates as f32*1.5) as usize);
    // Keep a flat list of keys to sample deletions in O(1) instead of iter().choose()
    let mut edge_keys: Vec<(usize, usize)> = Vec::with_capacity(num_updates * 2);
    let mut edge_index: HashMap<(usize, usize), usize> = HashMap::with_capacity(num_updates * 2);

    let mut add_edge_key = |key: (usize, usize), edge_keys: &mut Vec<(usize, usize)>, edge_index: &mut HashMap<(usize, usize), usize>| {
        let idx = edge_keys.len();
        edge_keys.push(key);
        edge_index.insert(key, idx);
    };

    let mut remove_edge_key = |key: (usize, usize), edge_keys: &mut Vec<(usize, usize)>, edge_index: &mut HashMap<(usize, usize), usize>| {
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
        match rng.random_range(0.0..1.0) < insert_prob{
            true => {
                // Insert an edge
                let u = rng.random_range(0..num_nodes);
                let v = rng.random_range(0..num_nodes);
                if u == v{
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

            },
            false => {
                // Delete an edge
                if !edge_keys.is_empty(){
                    let idx = rng.random_range(0..edge_keys.len());
                    let (u, v) = edge_keys[idx];
                    let w = *known_edges.get(&(u, v)).unwrap();
                    // with probability 0.5 delete the entire edge
                    if rng.random_range(0.0..1.0) < full_delete_prob{
                        operations.push(Instruction::Delete(counter,u, v));
                        known_edges.remove(&(u,v));
                        known_edges.remove(&(v,u));
                        remove_edge_key((u, v), &mut edge_keys, &mut edge_index);
                        remove_edge_key((v, u), &mut edge_keys, &mut edge_index);
                    }else{
                        // delete half the weight
                        let new_w: f64 = w/2.0;
                        operations.push(Instruction::DeleteHalf(counter,u, v, new_w));
                        // update the known edges
                        known_edges.entry((u,v)).and_modify(|e| *e -= new_w).or_insert(new_w);
                        known_edges.entry((v,u)).and_modify(|e| *e -= new_w).or_insert(new_w);
                    }
                }
            }
        }
        if rng.random_range(0.0..1.0) < increment_prob{
            counter += 1;
        }
    }
    pb.finish_with_message("Done generating instructions");
    GeneratedCommands{ nodes, operations }
}




#[test]
fn test_graph(){

}
