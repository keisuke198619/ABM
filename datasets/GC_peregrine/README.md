## Python codes for preprocessing of Granger causality (GC) of peregrine

* This is the python code for preprocessing of the following paper
* We addtionally analyzed peregrine data obtained in the following paper:
* Brighton, Caroline H.; Thomas, Adrian L. R.; Taylor, Graham K. (2018), Data from: Terminal attack trajectories of peregrine falcons are described by the proportional navigation guidance law of missiles, Dryad, Dataset, https://doi.org/10.5061/dryad.md268

* Note that results of the peregrine data are not shown in the following paper

### Author
* Keisuke Fujii - https://sites.google.com/view/keisuke1986en/
* Kazushi Tsutsui - https://researchmap.jp/ktsutsui?lang=en

### Reference
Keisuke Fujii, Naoya Takeishi, Kazushi Tsutsui, Emyo Fujioka, Nozomi Nishiumi, Ryoya Tanaka, Mika Fukushiro, Kaoru Ide, Hiroyoshi Kohno, Ken Yoda, Susumu Takahashi, Shizuko Hiryu, Yoshinobu Kawahara,  
Learning interaction rules from multi-animal trajectories via augmented behavioral models, 
Advances in Neural Information Processing Systems (NeurIPS'21), 34, 2021

### Preprocessing

* run `***` for preprocessing and saving `peregrine_data.npy`.
* The preprocessed data including only interaction were further analyzed. 
* for details, see `./preprocessing/preprocessing_peregrine_interaction.ipynb`.

### Main analysis and post analysis

* set this folder at `./datasets/GC_peregrine` in the folder of https://github.com/keisuke198619/ABM 

* for details, see https://github.com/keisuke198619/ABM
 
If you have any questions or requests, please contact me: fujii[at]i.nagoya-u.ac.jp
