## Learning interaction rules from multi-animal trajectories via augmented behavioral models (ABM)

### Requirements
* python 3
* To install requirements:

```setup
pip install -r requirements.txt
```


### Experiments

* see `run.sh` for commands using various datasets.

* The synthetic datasets and peregrine dataset are stored in the folder `./datasets`.

* Other animal data should be set in the folder `./datasets/GC_**`.

* Further details are documented within the code.

### Acknowledgements

Codes for the baseline models are not included into this project and is available in the following repositories:
- eSRU: https://github.com/sakhanna/SRU_for_GCI
- ACM: https://github.com/loeweX/AmortizedCausalDiscovery
- Linear GC and Local TF: https://github.com/tailintalent/causal
