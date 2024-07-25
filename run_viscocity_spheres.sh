#!/bin/sh

python experiments_spheres.py data/local/experiments/tonic/swimmer-swim_in_viscous_spheres/ncap_head_spheres/ results/viscocity_spheres/ncap_head_spheres.pth True
python experiments_spheres.py data/local/experiments/tonic/swimmer-swim_in_viscous_spheres/ncap_nohead_spheres/ results/viscocity_spheres/ncap_nohead_spheres.pth True

python experiments_spheres.py tonic/data/local/experiments/tonic/swimmer-swim/pretrained_mlp_ppo/ results/viscocity_spheres/pretrained_mlp_vis0.pth False
python experiments_spheres.py tonic/data/local/experiments/tonic/swimmer-swim/mlp_256_viscosity_0.1/ results/viscocity_spheres/pretrained_mlp_vis1.pth False
python experiments_spheres.py tonic/data/local/experiments/tonic/swimmer-swim/mlp_256_viscosity_0.2/ results/viscocity_spheres/pretrained_mlp_vis2.pth False
python experiments_spheres.py tonic/data/local/experiments/tonic/swimmer-swim/mlp_256_viscosity_0.4/ results/viscocity_spheres/pretrained_mlp_vis4.pth False

python experiments_spheres.py tonic/data/local/experiments/tonic/swimmer-swim/pretrained_ncap_ppo/ results/viscocity_spheres/pretrained_ncap_vis0.pth True
python experiments_spheres.py tonic/data/local/experiments/tonic/swimmer-swim/ncap_ppo_viscosivity_0.1/ results/viscocity_spheres/pretrained_ncap_vis1.pth True
python experiments_spheres.py tonic/data/local/experiments/tonic/swimmer-swim/ncap_ppo_viscosivity_0.2/ results/viscocity_spheres/pretrained_ncap_vis2.pth True
python experiments_spheres.py tonic/data/local/experiments/tonic/swimmer-swim/ncap_ppo_viscosivity_0.4/ results/viscocity_spheres/pretrained_ncap_vis4.pth True

python experiments_spheres.py tonic/data/local/experiments/tonic/swimmer-swim_vis0/mlp_ppo_vis0_10M/ results/viscocity_spheres/pretrained_mlp_vis0_10M.pth True
python experiments_spheres.py tonic/data/local/experiments/tonic/swimmer-swim_vis4/mlp_ppo_vis4_10M_notime/ results/viscocity_spheres/pretrained_mlp_vis4_10M.pth True