#!/bin/bash

# bats
# python run_grid_search.py --experiment bats --model gvar --K 3 --self_other --CF_pred --percept --test_samples 1 --TEST 

# peregrine
# python run_grid_search.py --experiment peregrine --model gvar --K 3 --self_other --CF_pred --percept --test_samples 1 --TEST 

# sula
# python run_grid_search.py --experiment sula --model gvar --K 3 --self_other --CF_pred --percept --test_samples 25 --TEST 

# mice
# python run_grid_search.py --experiment mice --model gvar --K 3 --TEST --data_dir ../ABM_data --test_samples 3
# python run_grid_search.py --experiment mice --model gvar --K 3 --self_other --CF_pred --percept --TEST --data_dir ../ABM_data --test_samples 2

# flies
# python run_grid_search.py --experiment flies --model gvar --K 3 --TEST --test_samples 2
# python run_grid_search.py --experiment flies --model gvar --K 3 --self_other --CF_pred --percept --TEST --test_samples 2 

# create kuramoto dataset: the length is a finally obtained length 
# cd datasets
# python generate_ODE_dataset.py --num_train 64 --num_valid 64 --num_test 64 --sample_freq 40 --length 200 --length_test 200 

# create boid dataset: 
# python generate_boid_dataset.py --num-train 10 --num-valid 10 --num-test 10 --sample_freq 100 --length 200 --length_test 200 --bat --partial --avoid #--video 10 

# kuramoto (VAR, GVAR/+self_other/+CF_pred)
# python run_grid_search.py --experiment kuramoto --model gvar --K 5 --TEST
# python run_grid_search.py --experiment kuramoto --model gvar --K 5 --self_other --TEST
# python run_grid_search.py --experiment kuramoto --model gvar --K 5 --self_other --CF_pred --TEST # 

# boid (GVAR/+self_other/+CF_pred/+percept)
# python run_grid_search.py --experiment boid5_bat --model gvar --K 3 --TEST
# python run_grid_search.py --experiment boid5_bat --model gvar --K 3 --self_other
# python run_grid_search.py --experiment boid5_bat --model gvar --K 3 --self_other --CF_pred 
# python run_grid_search.py --experiment boid5_bat --model gvar --K 3 --self_other --percept 
# python run_grid_search.py --experiment boid5_bat --model gvar --K 3 --self_other --CF_pred --percept --TEST #
