#!/bin/bash
cd /home/fujii/workspace4/work/ABM
source $HOME/workspace4/virtualenvs/cause/bin/activate 

# bats
python run_grid_search.py --experiment bats --model gvar --K 3 --test_samples 2 --self_other --CF_pred --percept --TEST --data_dir ../ABM_data 

# peregrine
python run_grid_search.py --experiment peregrine --model gvar --K 3 --self_other --CF_pred --percept --test_samples 2 --TEST 

# sula
python run_grid_search.py --experiment sula --model gvar --K 3 --self_other --CF_pred --percept --test_samples 2 --TEST --data_dir ../ABM_data

# mice
python run_grid_search.py --experiment mice --model gvar --K 3 --self_other --CF_pred --percept --test_samples 2 --TEST --data_dir ../ABM_data

# kuramoto (VAR, GVAR/+self_other/+CF_pred)
# python run_grid_search.py --experiment kuramoto --model gvar --K 5 
# python run_grid_search.py --experiment kuramoto --model gvar --K 5 --self_other --TEST
# python run_grid_search.py --experiment kuramoto --model gvar --K 5 --self_other --CF_pred # --TEST # 

# boid (GVAR/+self_other/+CF_pred/+percept)
# python run_grid_search.py --experiment boid5_bat --model gvar --K 3 --TEST
# python run_grid_search.py --experiment boid5_bat --model gvar --K 3 --self_other
# python run_grid_search.py --experiment boid5_bat --model gvar --K 3 --self_other --CF_pred 
# python run_grid_search.py --experiment boid5_bat --model gvar --K 3 --self_other --percept 
# python run_grid_search.py --experiment boid5_bat --model gvar --K 3 --self_other --CF_pred --percept --TEST #
