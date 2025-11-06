

use raphtory::prelude::*;
use raphtory::db::graph::views::deletion_graph::PersistentGraph;
use indicatif::ProgressBar;

use rayon::prelude::*;
use raphtory::core::entities::{EID, VID};
use raphtory::db::graph::edge::EdgeView;


#[derive(PartialEq, PartialOrd, Debug)]
pub enum EdgeOp{
    Update(VID,VID, f64), // Add/remove this amount (src,dst,weight)
    Delete(VID,VID) // Clear the contribution of this edge (src,dst)
}
#[derive(PartialEq, PartialOrd, Debug, Copy, Clone)]
pub enum TemporalOp{
    Update(i64, f64), // update to this amount (time, delta)
    Delete(i64) // Clear the contribution of this edge (time)
}

impl TemporalOp{
    pub fn time(&self)->i64{
        match self{
            TemporalOp::Update(x,_) =>*x,
            TemporalOp::Delete(x) => *x

        }
    }
}


/// For holding update diffs to be applied to an external data structure
/// Applying the updates in the first bucket syncs the data structure to the graph
/// at time times[0].
#[derive(Debug)]
pub struct Diffs{
    pub times: Vec<i64>,
    pub buckets: Vec<Vec<EdgeOp>>
}

impl Diffs{
    pub fn iter(&self) -> impl Iterator<Item = (&i64, &Vec<EdgeOp>)>{
        self.times.iter().zip(self.buckets.iter())
    }
}


/// Returns a list of (start,stop,prop_value) stop is exclusive
pub fn build_timeline(edge: EdgeView<&PersistentGraph>,prop: &str) -> Vec<(i64,i64, f64)>{
    // Extract a Vec of (start,stop,weight) representing the edge's history for property prop (assumed to be f64)
    edge.explode().into_iter().filter(|x| x.properties().get(prop).is_some()).map(|e|{
        (
            e.earliest_time().unwrap(),
            e.latest_time().unwrap(),
            e.properties().get(prop).unwrap().as_f64().unwrap()
        )
    }).collect()
}


/// Given an edge timeline, return a list of buckets where each bucket describes how to sync with a particular time.
pub fn bucket_timeline(timeline: &[(i64,i64, f64)], bucket_starts: &Vec<i64>) ->Vec<Vec<TemporalOp>>{
    

    let mut buckets = bucket_starts.iter().map(|_|vec!()).collect::<Vec<Vec<TemporalOp>>>();
    if bucket_starts.is_empty()|| bucket_starts.is_empty(){
        return vec![];
    }
    // First we build a list of events (update or delete) based on gaps in the timeline:    
    let mut events = vec!();

    for i in 1..(timeline.len()+1){
        if  i == timeline.len() || timeline[i].0 == timeline[i-1].1{
            // No gap. Previous was just an update:
            events.push(TemporalOp::Update(timeline[i-1].0,timeline[i-1].2));
        }else{
            // Gap implies deletion after previous:
            events.push(TemporalOp::Update(timeline[i-1].0,timeline[i-1].2));
            events.push(TemporalOp::Delete(timeline[i-1].1));
        }
    }
    // Then bucket these events:
    let mut bucket_idx = 0;
    for event in events{
        while bucket_idx+1 < bucket_starts.len() && event.time() >= bucket_starts[bucket_idx]{
            bucket_idx+=1
        }
        buckets[bucket_idx].push(event);
    }
    // Now we ignore all but the last update in each bucket:
    let mut buckets = buckets.into_iter().map(|mut bucket|{
        bucket.pop()
    }).collect::<Vec<_>>();

    // Finally, we convert to a diff. That is, replace absolute updates with relative updates, reset after deletions.
    let mut prev = None;
    
    for action in &mut buckets{
        match action{
            Some(TemporalOp::Update(_,x )) => {
                let diff = *x - prev.as_ref().unwrap_or(&0.0);
                *x = diff;
                *prev.get_or_insert(0.0) += diff;
            },
            Some(TemporalOp::Delete(_)) => {
                // ignore extra deletes:
                match prev{
                    Some(_) => prev = None,
                    None => *action = None,
                }
            },
            None => {}
        }
    }

    // Now swap actions to single element lists:
    buckets.into_iter().map(|x|{
        match x{
            Some(y) => vec!(y),
            None => vec!()
        }
    }).collect()

}



pub fn build_diff(graph: &PersistentGraph, start:i64, end: i64, step_size: usize, prop_name:&str) -> Diffs{
    // Process the edges in a persistent graph so diffs can be 
    // quickly applied by another data structure to sync with consecutive steps
    let bucket_starts = (start..end).step_by(step_size).skip(1).collect::<Vec<_>>();

    // for the final reduce:
    let identity = || bucket_starts.iter().map(|_| Vec::new()).collect::<Vec<Vec<EdgeOp>>>();


    let buckets = graph.edges().iter().par_bridge().map(|edge|{
        // get the bucket timeline for each edge, and spit out the corresponding actions
        let src = edge.edge.src();
        let dst = edge.edge.dst();
        let timeline = build_timeline(edge, prop_name);
        bucket_timeline(&timeline, &bucket_starts).into_iter().map(|buckets|{
            buckets.into_iter().map(|action|{
                match action{
                    TemporalOp::Update(t,x ) => EdgeOp::Update(src, dst, x),
                    TemporalOp::Delete(t) => EdgeOp::Delete(src, dst)
                }
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>()
    }).reduce(identity , |mut acc: Vec<Vec<EdgeOp>>,b: Vec<Vec<EdgeOp>>|{
        for (x,y) in acc.iter_mut().zip(b){
            x.extend(y);
        }
        acc
    });

    Diffs { times: bucket_starts, buckets }
}