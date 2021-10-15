## Python codes for preprocessing analysis of Granger causality (GC) of flies

This is the python code for preprocessing analysis of the following paper

### Author
* Keisuke Fujii - https://sites.google.com/view/keisuke1986en/
* Kazushi Tsutsui - https://researchmap.jp/ktsutsui?lang=en

### Reference
Keisuke Fujii, Naoya Takeishi, Kazushi Tsutsui, Emyo Fujioka, Nozomi Nishiumi, Ryoya Tanaka, Mika Fukushiro, Kaoru Ide, Hiroyoshi Kohno, Ken Yoda, Susumu Takahashi, Shizuko Hiryu, Yoshinobu Kawahara,  
Learning interaction rules from multi-animal trajectories via augmented behavioral models, 
Advances in Neural Information Processing Systems (NeurIPS'21), 34, 2021

### Preprocessing

* run `preprocessing/preprocessing_flies.ipynb` for preprocessing and saving `flies_data.npy`.
* The preprocessed data including only interaction were further analyzed. 
* for details, see `./preprocessing/preprocessing_flies_interaction.ipynb`.

### Main analysis and post analysis

* set this folder at `./datasets/GC_flies` in the folder of https://github.com/keisuke198619/ABM 

* for details, see https://github.com/keisuke198619/ABM
 
If you have any questions or requests, please contact me: fujii[at]i.nagoya-u.ac.jp
