use crate::{
    alg::TreeData,
    diff::{ExtendedEdgeOp, NodeOps},
    snapshot_clustering::{GraphLike, PartitionOutput, PartitionType, SnapshotClusteringAlg},
};
use core::time;
use faer::{
    sparse::{SparseRowMat, SymbolicSparseRowMat},
    traits::num_traits::Inv,
};
use itertools::{Itertools, multiunzip};
use rand::{Rng, SeedableRng, rngs::StdRng};
use raphtory::{algorithms::cores, db::graph::node};
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use anyhow::{Result, anyhow};
use rustc_hash::{FxHashMap, FxHashSet};

use super::DynamicClustering;
use super::common::*;
use rayon::prelude::*;

pub struct Coreset<V> {
    pub nodes: Vec<V>,
    pub node_indices: Vec<TreeIndex>,
    pub weights: Vec<Float>,
    pub coreset_labels: Option<Vec<usize>>,
}

// Holds info for coreset construction.
pub struct SamplingInfo<V> {
    pub x_star: V,
    pub sigma: Float,
    pub sigma_over_x_star_deg: Float,
    pub timestamp: usize,
    pub x_star_seed_set_volume_inv: Float,
    pub total_contribution_inv: Option<Contribution>,

    // store the weight of each seed
    seed_weight: FxHashMap<V, Float>,
    // lazy seed map for every node
    seed_map: FxHashMap<V, V>,
}

impl<V: std::hash::Hash + Eq + Clone + Copy> SamplingInfo<V> {
    pub fn new(
        x_star: V,
        sigma: Float,
        sigma_over_x_star_deg: Float,
        timestamp: usize,
        total_weight: Float,
    ) -> Self {
        let mut seed_weight = FxHashMap::<V, Float>::default();
        // Initially, the only seed is x^*, with seed weight equal to the total weight (volume) of the input
        seed_weight.insert(x_star.clone(), total_weight);

        let mut seed_map = FxHashMap::<V, V>::default();
        // Initially, we just have x^* maps to itself:
        seed_map.insert(x_star, x_star);

        Self {
            x_star,
            sigma,
            sigma_over_x_star_deg,
            timestamp,
            x_star_seed_set_volume_inv: Float::from(1.0) / total_weight,
            total_contribution_inv: None,
            seed_weight,
            seed_map,
        }
    }

    pub fn get_seed(&mut self, node: V) -> V {
        // return the seed of a point, defaulting to x_star if not seen before
        self.seed_map.entry(node).or_insert(self.x_star).clone()
    }

    pub fn set_seed(&mut self, node: V, seed: V) {
        // Overwrite any existing seed entry (get_seed initializes to x_star on first access).
        self.seed_map.insert(node, seed);
    }

    pub fn modify_seed_weight(&mut self, seed: V, diff: Float) {
        // increment the seed weight of seed by diff. If it is not present, insert it with this value
        let entry = self.seed_weight.entry(seed).or_insert(Float::from(0.0));
        *entry += diff;
        let new_weight = *entry;

        // keep x_star seed set volume in sync for g() smoothing term
        if seed == self.x_star {
            debug_assert!(
                new_weight > Float::from(0.0),
                "x_star seed set weight must stay positive"
            );
            self.x_star_seed_set_volume_inv = Float::from(1.0) / new_weight;
        }
    }

    pub fn get_seed_weight(&mut self, seed: V) -> Float {
        *self.seed_weight.get(&seed).unwrap()
    }
}

impl<const ARITY: usize, V: std::hash::Hash + Eq + Clone + Copy + Send + Sync>
    DynamicClustering<ARITY, V>
{
    pub fn extract_coreset(
        &mut self,
        graph: &impl GraphLike<V = V>,
        coreset_size: usize,
        sampling_seeds: usize,
        time: i64,
    ) -> Result<Coreset<V>> {
        // basic sanity: can't sample more seeds than leaves or build a coreset smaller than the seed set
        let root_size = self.tree_data.size.first().unwrap_or(&0);
        if !(sampling_seeds < coreset_size && coreset_size < *root_size) {
            return Err(anyhow!(
                "Expected sampling_seeds < coreset_size < size(root); got {} < {} < {} at time {}",
                sampling_seeds,
                coreset_size,
                root_size,
                time
            ));
        }

        // Increment internal (logical) timestamp:
        self.timestamp += 1;
        let timestamp = self.timestamp;
        let sigma = self.sigma;

        assert!(
            self.degrees.peek().unwrap().1.0 > Float::from(0.0),
            " Degree top deg node was zero"
        );

        let (&x_star, x_star_degree) = self.degrees.peek().ok_or(anyhow!("No nodes in graph"))?;

        debug_assert!(
            x_star_degree.0.is_finite() && *x_star_degree != NodeDegree(Float::from(0.0)),
            "x_star must have positive finite degree"
        );

        let sigma_over_x_star_deg = sigma * x_star_degree.inv().0;

        let total_weight = self.tree_data.volume[0].0;
        debug_assert!(
            total_weight.is_finite() && total_weight != Float::from(0.0),
            "total weight must be positive and finite"
        );

        let mut info = SamplingInfo::new(
            x_star,
            sigma,
            sigma_over_x_star_deg,
            timestamp,
            total_weight,
        );

        // sanity: leaves marked deleted should carry zero volume
        // self.assert_zero_volume_for_empty_leaves(&info);

        // first we add x_star:
        self.repair(x_star, &mut info, graph, time);

        // print the value of f for the whole tree and compare against the sum of the
        // per-child contributions computed by the fused array version. Note that
        // contribution_from_arrays returns the number of children filled, not the sum.
        let f = self.f(TreeIndex(0), &info);
        let mut buffer = [Float::from(0.0); ARITY];
        let filled = self.contribution_from_arrays(&mut buffer, TreeIndex(0), &info);
        let f_array = buffer[..filled].iter().copied().collect_vec();
        // println!("top f: {}, array_f_sum: {:?}", f, f_array);

        let mut rng = rand::rngs::StdRng::from_os_rng();

        // Now we sample a node uniformly:
        let tree_size = self.tree_data.size.len();
        let num_leaves = self.node_to_tree_map.len();

        let uniform_idx = TreeIndex(rng.random_range(tree_size - num_leaves..tree_size));
        let uniform_node = *self.tree_to_node_map.get(&uniform_idx).unwrap();
        self.repair(uniform_node, &mut info, graph, time);

        let remaining_seeds = sampling_seeds.saturating_sub(2);
        for i in 0..remaining_seeds {
            // Sample a point according to f:
            let (node, _, _) = self.sample(&info, &mut rng).map_err(|e| {
                anyhow!("failed sampling seed {} of {}: {e}", i + 1, remaining_seeds)
            })?;
            self.repair(node, &mut info, graph, time);
        }

        // populate total_contribution_inv
        let total_contribuiton = self.f(TreeIndex(0), &info);
        debug_assert!(
            total_contribuiton.0.is_finite()
                && total_contribuiton != Contribution(Float::from(0.0)),
            "total contribution must be positive and finite"
        );
        info.total_contribution_inv = Some(total_contribuiton.inv());

        let coreset_size_f = Float::from(coreset_size as Float_Dtype);
        let coreset_iterator = (0..coreset_size).map(|_| {
            let (node, idx, prob) = self.sample_smoothed(&info, &mut rng).unwrap();
            let node_deg = self.tree_data.volume[idx].0;
            let weight = node_deg / (prob * coreset_size_f);
            (node, idx, weight)
        });

        // Now we deduplicate the coreset:
        let mut coreset: FxHashMap<(V, TreeIndex), Float> = FxHashMap::default();
        for (v, index, weight) in coreset_iterator {
            *coreset.entry((v, index)).or_default() += weight;
        }

        let (unique_vs, unique_indices, weights): (Vec<_>, Vec<_>, Vec<_>) =
            multiunzip(coreset.into_iter().map(|((v, idx), w)| (v, idx, w)));

        Ok(Coreset {
            nodes: unique_vs,
            node_indices: unique_indices,
            weights,
            coreset_labels: None,
        })
    }

    pub fn repair(
        &mut self,
        point_added: V,
        info: &mut SamplingInfo<V>,
        graph: &impl GraphLike<V = V>,
        time: i64,
    ) {
        // We implicitly add the point to the init set, update it's neighbours,
        // and seed maps/ seed weight

        let point_added_index = *self.node_to_tree_map.get(&point_added).unwrap();

        let point_added_deg = self.tree_data.volume[point_added_index].0;
        let old_seed = info.get_seed(point_added);

        // subtract weight from the old seed and add it to the new seed (the point we're adding)
        info.modify_seed_weight(old_seed, -point_added_deg);
        info.modify_seed_weight(point_added, point_added_deg);
        // New point becomes its own seed. This and the above is a no-op for x^* itself.
        info.set_seed(point_added, point_added);

        // Now we zero the contribution of this point in the tree by setting the corresponding f_delta term to
        // the corresponding f_b term.
        // We also bump the timestamps
        let f_b = self.f_b(point_added_index, info);
        self.tree_data.f_delta[point_added_index] = FDelta(f_b);
        self.tree_data.timestamp[point_added_index] = info.timestamp;

        // Now update corresponding delta terms at internal nodes:
        self.apply_updates_from_single(point_added_index, |other, idx| {
            Self::one_step_recompute_with_timestamp(
                idx,
                &mut other.tree_data.f_delta,
                &other.tree_data.timestamp,
                info.timestamp,
                |_i| FDelta::zero(),
            );
            other.tree_data.timestamp[idx] = info.timestamp;
        });

        // used to track every seed set which lost members to the new set
        let mut old_seeds = FxHashSet::default();
        old_seeds.insert(old_seed);

        let neighbours = graph.neighbours(&point_added, time).collect::<Vec<_>>();

        // if point_added == info.x_star {
        //     println!("x_star has {} neighbours", neighbours.len());
        // }

        let mut filtered_neighbours = Vec::with_capacity(neighbours.len());

        for (neighbour, edge_weight) in neighbours.iter() {
            let neighbour_idx = *self.node_to_tree_map.get(neighbour).unwrap();
            let neighbour_deg = self.tree_data.volume[neighbour_idx].0;

            let weighted_distance_to_point_added = Self::weighted_kernel_distance(
                point_added_deg,
                EdgeWeight(Float::from(*edge_weight)),
            );
            let current_contribution = self.f(neighbour_idx, info);

            if weighted_distance_to_point_added < current_contribution {
                // neighbour is now closer to this point.
                filtered_neighbours.push(*neighbour);

                // We update it's delta term, bubble updates, and reconcile seed maps/weights.
                // We also bump timestamps
                let new_f_delta_term =
                    self.f_b(neighbour_idx, info) - weighted_distance_to_point_added.0;
                self.tree_data.f_delta[neighbour_idx] = FDelta(new_f_delta_term);
                self.tree_data.timestamp[neighbour_idx] = info.timestamp;

                self.apply_updates_from_single(neighbour_idx, |other, idx| {
                    Self::one_step_recompute_with_timestamp(
                        idx,
                        &mut other.tree_data.f_delta,
                        &other.tree_data.timestamp,
                        info.timestamp,
                        |_i| FDelta::zero(),
                    );
                    other.tree_data.timestamp[idx] = info.timestamp;
                });

                let old_seed = info.get_seed(*neighbour);
                old_seeds.insert(old_seed);
                info.modify_seed_weight(old_seed, -neighbour_deg);
                info.modify_seed_weight(point_added, neighbour_deg);
                info.set_seed(*neighbour, point_added);
            }
        }

        let seed_weight = info.get_seed_weight(point_added);

        for z in filtered_neighbours
            .into_iter()
            .chain([point_added].into_iter())
        {
            let z_idx = *self.node_to_tree_map.get(&z).unwrap();
            // set h_b to zero (if it's not already)
            self.tree_data.h_b[z_idx] = HB(Float::from(0.0));
            // set h_s to deg(z)/SeedWeight(point_added)
            let deg_z = self.tree_data.volume[z_idx].0;
            debug_assert!(
                seed_weight != Float::from(0.0),
                "seed weight must be non-zero for h_s update"
            );
            self.tree_data.h_s[z_idx] = HS(deg_z / seed_weight);

            // update timestamps
            self.tree_data.timestamp[z_idx] = info.timestamp;

            // trigger updates for h_b/h_s. Note we already have bumped the timestamps for these nodes previously
            // (both neighbours of point added adn point added itself)
            self.apply_updates_from_single(z_idx, |other, idx| {
                Self::one_step_recompute_with_timestamp(
                    idx,
                    &mut other.tree_data.h_b,
                    &other.tree_data.timestamp,
                    info.timestamp,
                    |i| convert(other.tree_data.volume[i]),
                );
                Self::one_step_recompute_with_timestamp(
                    idx,
                    &mut other.tree_data.h_s,
                    &other.tree_data.timestamp,
                    info.timestamp,
                    |_i| HS::zero(),
                );
                other.tree_data.timestamp[idx] = info.timestamp;
            });
        }

        // Now we update the h_s terms for nodes in old seed sets, based on changes in seed set weights, except for x^*

        // extract some stuff from info to avoid borrowing issues:
        let x_star = info.x_star;
        let timestamp = info.timestamp;

        let old_seeds_and_weights = old_seeds
            .into_iter()
            .filter(|s| *s != x_star)
            .map(|s| (s, info.get_seed_weight(s)))
            .collect::<Vec<_>>();

        let mut h_s_update_set = FxHashSet::default();

        for (s, seed_weight) in old_seeds_and_weights {
            debug_assert!(
                seed_weight != Float::from(0.0),
                "old seed weight must be non-zero for h_s rescale"
            );
            for z in graph.neighbours(&s, time).filter_map(|(neighbour, _)| {
                match info.get_seed(neighbour) == s {
                    true => Some(neighbour),
                    false => None,
                }
            }) {
                let z_idx = *self.node_to_tree_map.get(&z).unwrap();
                let deg_z = self.tree_data.volume[z_idx].0;
                self.tree_data.h_s[z_idx] = HS(deg_z / seed_weight);
                self.tree_data.timestamp[z_idx] = timestamp;
                h_s_update_set.insert(z_idx);
            }
        }

        // batch update the h_s terms for nodes in old seed sets:
        self.apply_updates_from_set(&h_s_update_set, |other, idx| {
            Self::one_step_recompute_with_timestamp(
                idx,
                &mut other.tree_data.h_s,
                &other.tree_data.timestamp,
                info.timestamp,
                |_i| HS::zero(),
            );
            other.tree_data.timestamp[idx] = info.timestamp;
        });
    }

    pub fn build_coreset_graph(
        &self,
        coreset: &Coreset<V>,
        time: i64,
        graph: &(impl GraphLike<V = V> + Sync),
    ) -> SparseRowMat<usize, f64> {
        let n = coreset.node_indices.len();
        let shift = self.sigma;

        let coreset_indices = &coreset.nodes;
        let coreset_weights = &coreset.weights;

        let degrees = coreset_indices
            .iter()
            .map(|idx| self.degrees.get(idx).unwrap().1)
            .collect::<Vec<_>>();

        let node_name_to_index = coreset_indices
            .iter()
            .enumerate()
            .map(|(idx, name)| (*name, idx))
            .collect::<FxHashMap<V, usize>>();

        let W_D_inv = (0..n)
            .map(|idx| coreset_weights[idx] / degrees[idx].0)
            .collect::<Vec<Float>>();

        // guess the number of non-zero entries in the coreset graph:
        let mut data = Vec::<f64>::with_capacity(n * 200);
        let mut indices = Vec::<usize>::with_capacity(n * 200);
        let mut indptr = Vec::<usize>::with_capacity(n + 1);
        let mut nnz_per_row = Vec::<usize>::with_capacity(n);

        let mut indptr_counter = 0;

        let neighbour_list = coreset_indices
            .as_slice()
            .par_iter()
            .map(|v| graph.neighbours(v, time).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        for (i, &node_name) in coreset_indices.iter().enumerate() {
            let neighbours = neighbour_list[i].iter();

            // get the neighbours of index that are in the coreset and transform the data
            // We are computing
            // A_C = W_CD^{-1}_C A_C D^{-1}_C W_C + W_C shift*D^{-1}_C W_C
            //     = W_CD^{-1}_C A_C D^{-1}_C W_C + shift* W_C*D^{-1}_C W_C
            // where:
            //  -A_C is the submatrix of A corresponding to the coreset indices,
            //  -W_C is the diagonal matrix of coreset weights,
            //  -D is the diagonal matrix of A and D_C is the submatrix of D corresponding to the coreset indices.
            let W_D_inv_i = W_D_inv[i];
            let mut no_diag = true;
            let mut good_indices_and_data_transformed = neighbours
                .filter_map(|(neighbour_name, data)| {
                    if node_name == *neighbour_name {
                        no_diag = false;
                        node_name_to_index.get(neighbour_name).map(|&coreset_j| {
                            (
                                coreset_j,
                                Float::from(*data) * W_D_inv_i * W_D_inv_i
                                    + shift * (coreset_weights[i]) * W_D_inv_i,
                            )
                        })
                    } else {
                        node_name_to_index.get(neighbour_name).map(|&coreset_j| {
                            (
                                coreset_j,
                                Float::from(*data) * W_D_inv_i * W_D_inv[coreset_j],
                            )
                        })
                    }
                })
                .collect::<Vec<(usize, Float)>>();

            if no_diag {
                // If no diagonal term present already, we add the diagonal term with just sigma W_CD^{-1} W_C:
                good_indices_and_data_transformed
                    .push((i, shift * (coreset_weights[i].0) * W_D_inv_i));
            }

            good_indices_and_data_transformed.sort_unstable_by_key(|&(idx, _)| idx);

            // push the data and indices to the data and indices vectors:
            data.extend(good_indices_and_data_transformed.iter().map(|x| x.1.0));
            indices.extend(good_indices_and_data_transformed.iter().map(|x| x.0));
            let nnz = good_indices_and_data_transformed.len();
            nnz_per_row.push(nnz);
            // push the indptr counter to the indptr vector and bump by nnz
            indptr.push(indptr_counter);
            indptr_counter += nnz;
        }
        // push the last indptr counter to the indptr vector
        indptr.push(indptr_counter);
        SparseRowMat::new(
            SymbolicSparseRowMat::<usize>::new_checked(n, n, indptr, Some(nnz_per_row), indices),
            data,
        )
    }

    pub fn rust_label_full_graph(
        &self,
        coreset: &Coreset<V>,
        num_clusters: usize,
        time: i64,
        graph: &(impl GraphLike<V = V> + Sync),
        nodes: Option<&[V]>,
    ) -> (Vec<V>, Vec<usize>, Vec<Float_Dtype>) {
        let shift = self.sigma;
        let coreset_names = coreset.nodes.clone();
        let coreset_weights = reinterpret_slice::<Float, f64>(coreset.weights.as_slice()).to_vec();
        let coreset_labels = coreset
            .coreset_labels
            .as_ref()
            .expect("coreset labels must be set before full-graph labelling");

        let node_names = match nodes {
            Some(subset) => subset.to_vec(),
            None => graph.nodes(time),
        };

        // Union of all nodes we will touch (labelled nodes + coreset) to deduplicate neighbour lookups.
        let mut all_nodes_set: FxHashSet<V> = node_names.iter().copied().collect();
        all_nodes_set.extend(coreset_names.iter().copied());
        let all_nodes: Vec<V> = all_nodes_set.iter().copied().collect();

        // Precompute degree lookups to avoid touching the priority queue in parallel.
        let degree_map: FxHashMap<V, Float> = all_nodes
            .iter()
            .map(|v| (*v, self.degrees.get(v).unwrap().1.0))
            .collect();

        // Precompute neighbourhoods for all nodes we will inspect (coreset + label targets).

        // Slowest part by far:
        // let t0 = std::time::Instant::now();
        let adjacency_vec = all_nodes
            .par_iter()
            .map(|node| (*node, graph.neighbours(node, time).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let adjacency: FxHashMap<_, _> = adjacency_vec.into_iter().collect();
        // println!("Took {} ms", t0.elapsed().as_millis());

        // Group the coreset nodes/weights by cluster label.
        let coreset_grouped = coreset_names
            .iter()
            .zip(coreset_labels.iter())
            .zip(coreset_weights.iter())
            .fold(
                vec![(Vec::new(), Vec::new()); num_clusters],
                |mut acc, ((&name, &label), &weight)| {
                    acc[label].0.push(name);
                    acc[label].1.push(weight);
                    acc
                },
            );

        // Compute center norms and denominators per cluster (parallel per center).
        let result = coreset_grouped
            .into_par_iter()
            .map(|(indices, weights)| {
                if indices.is_empty() {
                    // Empty cluster: give it an infinite norm so it is never chosen as the default.
                    return (Float::from(f64::INFINITY), Float::from(0.0));
                }

                let indices_set: FxHashSet<V> = indices.iter().copied().collect();
                let index_to_weight: FxHashMap<V, Float_Dtype> = indices
                    .iter()
                    .copied()
                    .zip(weights.iter().copied())
                    .collect();

                let denom = Float::from(weights.iter().sum::<Float_Dtype>());
                // Defensive: avoid zero denom
                if denom == Float::from(0.0) {
                    return (Float::from(f64::INFINITY), Float::from(0.0));
                }
                let mut center_norm_sum = Float::from(0.0);

                indices.iter().for_each(|i| {
                    let weight = index_to_weight[i];
                    let vertex_degree =
                        *degree_map.get(i).expect("degree missing for coreset node");
                    let neighbours = adjacency.get(i).map(|x| x.as_slice()).unwrap_or(&[]);

                    let neighbour_contrib =
                        neighbours.iter().fold(Float::from(0.0), |acc, (j, v)| {
                            if indices_set.contains(j) {
                                let neighbour_degree =
                                    *degree_map.get(j).expect("degree missing for neighbour");
                                let value = if i != j {
                                    Float::from(*v) / (vertex_degree * neighbour_degree)
                                } else {
                                    Float::from(*v) / (vertex_degree * neighbour_degree)
                                        + shift / vertex_degree
                                };
                                acc + value * Float::from(weight) * Float::from(index_to_weight[j])
                            } else {
                                acc
                            }
                        });

                    center_norm_sum += neighbour_contrib;
                });

                center_norm_sum /= denom * denom;
                (center_norm_sum, denom)
            })
            .collect::<Vec<(Float, Float)>>();

        let (center_norms, center_denoms): (Vec<Float>, Vec<Float>) = result.into_iter().unzip();

        // Pick the smallest finite center norm; if none are finite, fall back to cluster 0.
        let mut smallest_center_by_norm = 0usize;
        let mut smallest_center_by_norm_value = Float::from(f64::INFINITY);
        for (idx, norm) in center_norms.iter().enumerate() {
            if norm.is_finite() && *norm < smallest_center_by_norm_value {
                smallest_center_by_norm = idx;
                smallest_center_by_norm_value = *norm;
            }
        }

        let coreset_set: FxHashSet<V> = coreset_names.iter().copied().collect();
        let label_map: FxHashMap<V, usize> = coreset_names
            .iter()
            .copied()
            .zip(coreset_labels.iter().copied())
            .collect();
        let weight_map: FxHashMap<V, Float_Dtype> = coreset_names
            .iter()
            .copied()
            .zip(coreset_weights.iter().copied())
            .collect();

        let labels_and_distances: (Vec<usize>, Vec<Float_Dtype>) = node_names
            .par_iter()
            .map(|i| {
                let vertex_degree = *degree_map
                    .get(i)
                    .expect("degree missing for node in labelling pass");
                let mut x_to_c_is: FxHashMap<usize, Float> = FxHashMap::default();

                if let Some(neighbours) = adjacency.get(i) {
                    neighbours.iter().for_each(|(indx, weight)| {
                        if coreset_set.contains(&indx) {
                            let label = label_map[&indx];
                            let neighbour_weight = weight_map[&indx];
                            let neighbour_degree = *degree_map
                                .get(&indx)
                                .expect("degree missing for neighbour in labelling pass");

                            let inner_prod_with_vertex = if i != indx {
                                Float::from(*weight) / (vertex_degree * neighbour_degree)
                            } else {
                                Float::from(*weight) / (vertex_degree * neighbour_degree)
                                    + shift / vertex_degree
                            };

                            x_to_c_is
                                .entry(label)
                                .and_modify(|e| {
                                    *e += Float::from(neighbour_weight) * inner_prod_with_vertex;
                                })
                                .or_insert(Float::from(neighbour_weight) * inner_prod_with_vertex);
                        }
                    });
                }

                x_to_c_is.iter_mut().for_each(|(k, v)| {
                    let denom = center_denoms[*k];
                    if denom.is_finite() && denom != Float::from(0.0) {
                        *v /= denom;
                    } else {
                        *v = Float::from(0.0);
                    }
                });

                let mut best_center_value = smallest_center_by_norm_value;
                let mut best_center = smallest_center_by_norm;

                x_to_c_is
                    .iter()
                    .filter(|(center, _)| center_norms[**center].is_finite())
                    .for_each(|(center, v)| {
                        let distance = center_norms[*center] - Float::from(2.0) * *v;
                        if distance < best_center_value {
                            best_center = *center;
                            best_center_value = distance;
                        }
                    });

                (
                    best_center,
                    (best_center_value
                        + Float::from(1.0) / (vertex_degree * vertex_degree)
                        + shift / vertex_degree)
                        .0,
                )
            })
            .unzip();

        (node_names, labels_and_distances.0, labels_and_distances.1)
    }
}
