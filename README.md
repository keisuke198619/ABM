## Learning interaction rules from multi-animal trajectories via augmented behavioral models (ABM)

* Draft: https://ja.overleaf.com/read/kvzpzmytdxzy

### Requirements
* python 3
* To install requirements:

```setup
pip install -r requirements.txt
```
### Preprocessing 

* The synthetic, bats, sula, flies, and peregrine datasets are stored in the folder `./datasets`.
* These can be preprocessed by the code in the folder `./datasets`.
* Other animal data should be set in the folder `./datasets/GC_**`.

### Main analysis

* see `run.sh` for commands using various datasets.
* Further details are documented within the code.

### Post analysis

* The post analysis was performed by matlab code in the folder `./matlab_post_analysis`.

### References

Codes for the baseline models are available in the following repositories:

- GVAR: https://openreview.net/forum?id=DEa4JdMWRHp
- eSRU: https://github.com/sakhanna/SRU_for_GCI
- ACD: https://github.com/loeweX/AmortizedCausalDiscovery
- Linear GC and Local TF: https://github.com/tailintalent/causal
