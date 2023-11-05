## Learning interaction rules from multi-animal trajectories via augmented behavioral models (ABM)

### Author
Keisuke Fujii - https://sites.google.com/view/keisuke1986en/

### Reference
Keisuke Fujii, Naoya Takeishi, Kazushi Tsutsui, Emyo Fujioka, Nozomi Nishiumi, Ryoya Tanaka, Mika Fukushiro, Kaoru Ide, Hiroyoshi Kohno, Ken Yoda, Susumu Takahashi, Shizuko Hiryu, Yoshinobu Kawahara,  
Learning interaction rules from multi-animal trajectories via augmented behavioral models, 
Advances in Neural Information Processing Systems (NeurIPS'21), 34, 2021 [Link](https://proceedings.neurips.cc/paper/2021/hash/5c572eca050594c7bc3c36e7e8ab9550-Abstract.html "NeurIPS 2021")

### Requirements
* Python 3.8
* To install requirements:

```setup
pip install -r requirements.txt
```

* For Python 3.6, see `requirements36.txt`

### Preprocessing 

* The synthetic, sula, flies, and peregrine datasets are stored in the folder `./datasets`.
* These can be preprocessed by the code in the folder `./datasets`.
* The output file **_data.npy includes the data in the form such that [files][agents, xy(z), timestamps].
* Other animal data can be set in the folder `./datasets/GC_**`.
* We addtionally analyzed peregrine data obtained at https://doi.org/10.5061/dryad.md268.

### Main analysis

* See `run.sh` for commands using various datasets.
* The output file is in the folder `./weights`.
* Further details are documented within the code.

### Post analysis

* The post analysis was performed by matlab code in the folder `./matlab_post_analysis`.
* (2023/11) The post analysis code by python is released as `post_analysis.py`. Currently, mice and flies data can be used (and videos are not generated). For example, run `python post_analysis.py --experiment mice --model gvar --K 3 --test_samples 2` (see also `run.sh`). 

### Note for using your own data

* The performance of this method will be better in relatively fewer and sparse agents with than those in more and dense agents. 

### References

Codes for the baseline models are available in the following repositories:

- GVAR: https://openreview.net/forum?id=DEa4JdMWRHp
- eSRU: https://github.com/sakhanna/SRU_for_GCI
- ACD: https://github.com/loeweX/AmortizedCausalDiscovery
- Linear GC and Local TF: https://github.com/tailintalent/causal
