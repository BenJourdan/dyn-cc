use faer::traits::num_traits::real::{Real};
use faer::traits::num_traits::Inv;

use itertools::izip;

use super::common::*;
use super::coreset_impls::SamplingInfo;
use super::DynamicClustering;

use anyhow::{
    Result,
    anyhow
};

pub struct Coreset<V>{
    pub nodes: Vec<V>,
    pub weights: Vec<f64>,
}


impl<const ARITY: usize, V: std::hash::Hash + Eq + Clone + Copy> DynamicClustering<ARITY, V> {


    #[inline(always)]
    pub fn f_b(&self, node_idx: TreeIndex, info: &SamplingInfo) -> Float{
        // f_b = sigma* size + (sigma * vol)/w(C(x^*))

        let size = Float::from(self.tree_data.size[node_idx] as f64);
        let vol = self.tree_data.volume[node_idx].0;
        let x_star_seed_set_volume_inv = info.x_star_seed_set_volume_inv;
        let sigma = info.sigma;
        sigma.mul_add(size, sigma*x_star_seed_set_volume_inv*vol)
    }

    #[inline(always)]
    pub fn f_delta_read(&self, node_idx: TreeIndex, info: &SamplingInfo) -> FDelta{
        // return saved f_delta if timestamps match, else return 0.
        let saved_timestamp = self.tree_data.timestamp[node_idx];
        let cur_timestamp = info.time_stamp;
        let saved_f_delta = self.tree_data.f_delta[node_idx];
        saved_f_delta * Float::from((saved_timestamp == cur_timestamp) as u8)
    }

    #[inline(always)]
    pub fn f(&self, node_idx: TreeIndex, info: &SamplingInfo) -> Contribution{
        // f_s = f_b - f_delta

        let f_b = self.f_b(node_idx, info);
        let f_delta = self.f_delta_read(node_idx, info);
        Contribution(f_b - f_delta.0)
    }

    #[inline(always)]
    pub fn contribution_from_arrays(
        // A fused version of f() to compute contributions for all children of a parent node
        // and write them into an output buffer.
        &self,
        output_buffer: &mut [Float; ARITY],
        parent_idx: TreeIndex,
        info: &SamplingInfo,
    )-> usize{
        let start = self.child_index(parent_idx, 0).0;
        let end = (start+ARITY).min(self.tree_data.size.len());

        let sizes = &self.tree_data.size[start..end];
        let volumes = &self.tree_data.volume[start..end];
        let saved_f_deltas = &self.tree_data.f_delta[start..end];
        let saved_timestamps = &self.tree_data.timestamp[start..end];

        let cur_timestamp = info.time_stamp;
        let x_star_seed_set_volume_inv = info.x_star_seed_set_volume_inv;
        let sigma = info.sigma;

        let filled = end - start;

        // clear output buffer in the unused portion:
        output_buffer[filled..].fill(Float::from(0.0));

        for (o, s, v, f_del, t) in izip!(
            &mut output_buffer[..filled], 
            sizes, 
            volumes, 
            saved_f_deltas, 
            saved_timestamps){
            let size_f = Float::from(*s as f64);
            let vol_f = v.0;
            let f_delta_f =  f_del.0 * Float::from((*t == cur_timestamp) as u8);

            let total = sigma.mul_add(
                size_f,
                sigma.mul_add(
                    x_star_seed_set_volume_inv*vol_f,
                    -f_delta_f
                )
            );
            *o = Real::max(total,Float::from(0.0));
        }
        filled
    }


    #[inline(always)]
    pub fn h_b(&self, node_idx: TreeIndex, info: &SamplingInfo) -> HB{
        let saved_timestamp = self.tree_data.timestamp[node_idx];
        let cur_timestamp = info.time_stamp;
        let saved_h_b = self.tree_data.h_b[node_idx];
        let vol = self.tree_data.volume[node_idx].0;

        // If timestamps match, return saved h_b. Else, return vol.
        if saved_timestamp == cur_timestamp{
            saved_h_b
        } else {
            HB(vol)
        }
    }

    #[inline(always)]
    pub fn h_s(&self, node_idx: TreeIndex, info: &SamplingInfo) -> HS{
        let saved_timestamp = self.tree_data.timestamp[node_idx];
        let cur_timestamp = info.time_stamp;
        let saved_h_s = self.tree_data.h_s[node_idx];

        // If timestamps match, return saved h_s. Else, return 0.
        if saved_timestamp == cur_timestamp{
            saved_h_s
        } else {
            HS(0.0.into())
        }
    }

    #[inline(always)]
    pub fn g(&self, node_idx: TreeIndex, info: &SamplingInfo) -> SmoothedContribution{
        // g = f(S)/f(X) + h_b(S)/w(C(x^*)) + h_s(S)

        let f_s = self.f(node_idx, info).0;
        let total_contribution_inv = info.total_contribution_inv.0;
        let x_star_seed_set_volume_inv = info.x_star_seed_set_volume_inv;
        let h_b = self.h_b(node_idx, info).0;
        let h_s = self.h_s(node_idx, info).0;

        SmoothedContribution(
            f_s.mul_add(
                total_contribution_inv,
                h_b.mul_add(
                    x_star_seed_set_volume_inv,
                    h_s
                )
            )
        )
    }


    #[inline(always)]
    pub fn smoothed_contribution_from_arrays(
        // A fused version of g() to compute smoothed contributions for all children of a parent node
        // and write them into an output buffer.
        &self,
        output_buffer: &mut [Float; ARITY],
        parent_idx: TreeIndex,
        info: &SamplingInfo,
    ) -> usize{
        let start = self.child_index(parent_idx, 0).0;
        let end = (start+ARITY).min(self.tree_data.size.len());

        let sizes = &self.tree_data.size[start..end];
        let volumes = &self.tree_data.volume[start..end];
        let saved_f_deltas = &self.tree_data.f_delta[start..end];
        let saved_h_bs = &self.tree_data.h_b[start..end];
        let saved_h_ss = &self.tree_data.h_s[start..end];
        let saved_timestamps = &self.tree_data.timestamp[start..end];

        let cur_timestamp = info.time_stamp;
        let x_star_seed_set_volume_inv = info.x_star_seed_set_volume_inv;
        let total_contribution_inv = info.total_contribution_inv.0;
        let sigma = info.sigma;

        let filled = end - start;

        // clear output buffer in the unused portion:
        output_buffer[filled..].fill(Float::from(0.0));

        for (o, s, v, f_del, h_b, h_s, t) in izip!(
            &mut output_buffer[..filled], 
            sizes, 
            volumes, 
            saved_f_deltas, 
            saved_h_bs,
            saved_h_ss,
            saved_timestamps){
            let size_f = Float::from(*s as f64);
            let vol_f = v.0;
            let f_delta_f =  f_del.0 * Float::from((*t == cur_timestamp) as u8);
            let h_b_f = if *t == cur_timestamp { h_b.0 } else { vol_f };
            let h_s_f = if *t == cur_timestamp { h_s.0 } else { Float::from(0.0) };

            let f_s = sigma.mul_add(
                size_f,
                sigma.mul_add(
                    x_star_seed_set_volume_inv*vol_f,
                    -f_delta_f
                )
            );

            let total = f_s.mul_add(
                total_contribution_inv,
                h_b_f.mul_add(
                    x_star_seed_set_volume_inv,
                    h_s_f
                )
            );
            *o = Real::max(total,Float::from(0.0));
        }
        filled
    }

    
    fn sample_impl(
        &mut self,
        info: &SamplingInfo,
        rng: &mut impl rand::Rng,
        fill: impl Fn(&Self, &mut [Float; ARITY], TreeIndex, &SamplingInfo) -> usize,
    ) -> Result<(V, Float)>{
        if self.tree_data.size.is_empty(){
            return Err(anyhow!("Cannot sample from an empty tree."));
        }

        let mut cur = TreeIndex(0);
        let mut prob = Float::from(1.0f64);
        let mut buffer = [Float::from(0.0f64); ARITY];
        let mut cdf_buffer = [Float::from(0.0f64); ARITY];

        while self.tree_data.size[cur] > 1{

            // cur corresponds to an internal node
            
            // populate buffer with contributions of children
            let filled = fill(self, &mut buffer, cur, info);

            let child_contribution_sum: Float = buffer[..filled].iter().sum();
            if child_contribution_sum == Float::from(0.0f64){
                return Err(anyhow!("Cannot sample from a tree with zero total contribution."));
            }
            let sample = rng.random_range(0.0..child_contribution_sum.0);
            cdf_buffer[..filled].copy_from_slice(&buffer[..filled]);

            for i in 1..filled{
                cdf_buffer[i] += cdf_buffer[i-1];
            }

            // Now we sample a child
            let child_idx = cdf_buffer[..filled]
                .iter()
                .position(|&x| x.0 >= sample)
                .ok_or(anyhow!("Failed to sample a child node."))?;
            prob *= buffer[child_idx] / child_contribution_sum;
            cur = self.child_index(cur, child_idx);
        }

        let node_id = self.tree_to_node_map.get(&cur).unwrap();
        Ok((*node_id, prob))
    }

    pub fn sample(
        &mut self,
        info: &SamplingInfo,
        rng: &mut impl rand::Rng
    ) -> Result<(V, Float)>{
        self.sample_impl(info, rng, |this, buf, parent, info| {
            this.contribution_from_arrays(buf, parent, info)
        })
    }

    pub fn sample_smoothed(
        &mut self,
        info: &SamplingInfo,
        rng: &mut impl rand::Rng
    ) -> Result<(V, Float)>{
        self.sample_impl(info, rng, |this, buf, parent, info| {
            this.smoothed_contribution_from_arrays(buf, parent, info)
        })
    }


    #[inline(always)]
    pub fn weighted_kernel_distance(deg_v: Float, w: EdgeWeight) -> Contribution{
        // get the contribution of u w.r.t v.
        // If v is being added, this is for computing the updated contribution of u.
        // w Delta(u,v) = w(u,v)/ deg(v)
        (
            w.0* deg_v.inv()
        ).into()
        }


}
