import argparse
import os
import numpy as np
import time
import copy
from datetime import date
from scipy import io
from experimental_utils import run_grid_search, eval_causal_structure, eval_causal_structure_binary

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

# Keisuke Fujii, 2021
# modifying the work and code: 
#   Interpretable Models for Granger Causality Using Self-explaining Neural Networks 
#   Ričards Marcinkevičs, Julia E Volgae, https://openreview.net/forum?id=DEa4JdMWRHp

parser = argparse.ArgumentParser(description='Grid search')


# Experiment
parser.add_argument('--experiment', type=str, default="kuramoto", help="Experiment to be performed (default: 'kuramoto')")
parser.add_argument('--data_dir', type=str, default="./datasets", help="Experiment to be performed (default: 'kuramoto')")

# Model specification
parser.add_argument('--model', type=str, default='gvar', help="Model to train (default: 'gvar')")
parser.add_argument('--K', type=int, default=5, help='Model order (default: 5)')
parser.add_argument('--num-hidden-layers', type=int, default=2, help='Number of hidden layers (default: 2)')
parser.add_argument('--hidden-layer-size', type=int, default=50, help='Number of units in the hidden layer (default: 50)')

# Training procedure
parser.add_argument('--batch-size', type=int, default=256, help='Mini-batch size (default: 256)')
parser.add_argument('--num-epochs', type=int, default=500, help='Number of epochs to train (default: 10)')
parser.add_argument('--initial-lr', type=float, default=0.0001, help='Initial learning rate (default: 0.0001)')
parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1 value for the Adam optimiser (default: 0.9)')
parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2 value for the Adam optimiser (default: 0.999)')

# Meta
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--num-sim', type=int, default=10, help='Number of simulations (default: 1)')
parser.add_argument('--use-cuda', type=bool, default=True, help='Use GPU? (default: true)')

parser.add_argument('--CF_pred', action='store_true', help='Theory-guided (previously, counterfactual) prediction in the paper')
parser.add_argument('--percept', action='store_true', help='Navigation function in the paper')
parser.add_argument('--self_other', action='store_true', help='use interaction information')
parser.add_argument('--TEST', action='store_true')
parser.add_argument('--dynamic_edge', action='store_true')
parser.add_argument("--test_samples", type=int, default=10)
parser.add_argument("--numProcess", type=int, default=16) 
# Parsing args
args = parser.parse_args()
# args.numProcess = 16
os.environ["OMP_NUM_THREADS"]=str(args.numProcess) 

# other experimental conditions
if not args.self_other: # GVAR
    args.bidirection = True
else: 
    args.bidirection = False

signed_structures = None
if args.CF_pred or args.percept:
    args.self_other = True 

if args.experiment == "kuramoto":
    args.navigation = False
    args.realdata = False
elif "boid" in args.experiment:
    args.navigation = True
    args.realdata = False
else:
    args.navigation = True
    args.realdata = True

if not args.navigation:
    args.percept = False
 
print( str(args.experiment) + " datasets...")

# input lists
datasets = [] ; structures = []
datasets_val = [] ; structures_val = []
datasets_te = [] ; structures_te = []
lambdas = None
gammas = None
dynamic_structures = None; dynamic_structures_val = None; dynamic_structures_te = None
index_None = None; index_None_val = None; index_None_te = None
args_list = None; args_list_val = None; args_list_te = None

def animal_data(args,loc,vel,len_samples=1,edges=None,edges_res=None,index_none=None,file_no=0):
    datasets = []
    structures = []; dynamic_structures = []
    out_dims = args.out_dims
    
    if "boid" in args.experiment:
        loc = loc[:len_samples]
        vel = vel[:len_samples]
    else:
        num_timesteps, all_dim, num_atoms = loc.shape
        ns = 1
        loc = loc.reshape((1,num_timesteps, all_dim, num_atoms))
        vel = vel.reshape((1,num_timesteps, all_dim, num_atoms))

    args.vel_max = np.nanmax(np.linalg.norm(vel,axis=2,ord=2))

    if args.self_other: 
        feat = np.concatenate([vel/args.vel_max,loc/args.max],2)
    else:
        feat = loc/args.max
    args.d_self = args.out_dims *2
    ns, num_timesteps, all_dim, num_atoms = feat.shape

    ns = len_samples
    K = num_atoms
    pred_len = args.T - 1 

    # variable for CF_pred
    half_distance = 1e-2
    sigma_kernel = (half_distance**2)/(2*np.log(2)) # 0.5 when normalized distance is half_distance
    args.sigma_kernel = sigma_kernel
    if args.self_other: 
        for i in range(K):
            idx = np.arange(K)
            idx = np.delete(idx,i)
            loc_ = loc[:,:,:,idx] - loc[:,:,:,i:i+1].repeat(K-1,3)
            vel_ = vel[:,:,:,i:i+1].repeat(K-1,3)/args.vel_max
            dist = np.linalg.norm(loc_,axis=2,ord=2)
            loc_ /= np.expand_dims(dist,2).repeat(out_dims,2)
            for ii in range(K-1):
                loc_ii = loc_[0,:,:,ii]
                if np.sum(np.isnan(loc_ii))>0:
                    idx2 = np.where(np.isnan(loc_ii[:,0])) 
                    loc_[0,idx2,:,ii] = np.zeros((len(idx2),loc_ii.shape[1]))

            vel__ = np.expand_dims(np.sum(vel_*loc_,axis=2),3) # if > 0, approach; elif < 0, separate
            loc_ = np.expand_dims(loc_.transpose((0,1,3,2)).reshape((ns, num_timesteps, (K-1)*out_dims)),3)
            #vec_ = np.concatenate([loc_,np.expand_dims(dist,3)],2)
            reciprocal = 1/(np.expand_dims(dist,3)) # reciprocal of distance 
            ind_rec = (reciprocal>2).nonzero() # *args.max
            reciprocal[ind_rec] = 2 # *args.max # upper limit (dist = 0.5 if dist < 0.5)
            vec_ = np.concatenate([loc_,vel__,reciprocal],2) # direction((K-1)*2), approach(K-1), reciprocal of distance(K-1)
            
            #gausskernel = np.expand_dims(np.exp(-(dist)**2/(2*sigma_kernel)),3) # np.exp(-1**2/(2*sigma_kernel)) 1->0.92, 2-> 0.73, 5->0.14
            #gausskernels = np.repeat(gausskernel,2,axis=3).reshape((ns, num_timesteps,(K-1)*2,1))
            #vec_ = np.concatenate([loc_*gausskernels,gausskernel],2)
            if index_none is not None:
                time_idx = np.arange(num_timesteps)
                idx_none = time_idx[np.where(index_none[:,i]==1)]
                vec_[:,idx_none,:,:] = np.ones((ns, np.sum(index_none[:,i]), (K-1)*all_dim//2+(K-1)*2,1))*1e-10
                
            vec = np.concatenate([vec,vec_],3) if i > 0 else vec_

        feat = np.concatenate([feat,vec],2) 
    num_dims = feat.shape[2]

    # feat: (vel*2,loc*2,[dir*2,signed_vel,dist]*(p-1))*p

    if "boid" in args.experiment and not args.dynamic_edge:
        # edges_ = np.zeros((ns,K,K)) 
        edges_= edges[:len_samples]
        for i in range(K):
            for s in range(len_samples):
                for j in range(K):
                    if np.sum(np.abs(edges_res[s,args.K:,i,j]))==0:
                        edges_[s,i,j] = 0

    args.num_dims = num_dims
    
    feat = feat[:,:num_timesteps,:,:].transpose((0,1,3,2)).reshape(ns,num_timesteps,num_dims*K) # reshape(K,dim): dim->K 

    # unnormalize: (1 + feat[:,:,0]) * (self._max - self._min)/2 + self._min

    for i in range(ns):
        datasets.append(feat[i])
        
        if "boid" in args.experiment and not args.dynamic_edge:
            structures.append(edges_[i])
            dynamic_structures.append(edges_res[i])
        elif edges is not None:
            structures.append(edges[i])
            dynamic_structures.append(None)
        else: 
            structures.append(None)
            dynamic_structures.append(None)

    return datasets, structures, args, dynamic_structures

def boid_data(args,suffix,train=1):
    if train == 0:
        str_data = "_valid"
        len_samples = args.test_samples
    else:
        str_data = "_test"
        len_samples = args.test_samples
    # max_min
    loc = np.load(os.path.join(args.data_dir, 'loc' + str_data + suffix + ".npy"))
    vel = np.load(os.path.join(args.data_dir, 'vel' + str_data + suffix + ".npy"))
    if "boid" in args.experiment:
        outs = np.load(os.path.join(args.data_dir, 'edges' + str_data + suffix + ".npz"))
        edges_res = outs[outs.files[0]] 
        edges = outs[outs.files[1]] 
    else: 
        edges = np.load(os.path.join(args.data_dir, 'edges' + str_data + suffix + ".npy"))

    datasets, structures, args, dynamic_structures = animal_data(args,loc,vel,len_samples,edges,edges_res)

    return datasets, structures, args, dynamic_structures

# Generate data 
if args.realdata:
    index_None_val = []; index_None_te = []
    args_list_val = []; args_list_te = []
    signed_structures_val = [] ; signed_structures_te = []
    dynamic_structures_val = []; dynamic_structures_te = []
    test_samples = range(args.test_samples)
    if args.experiment == "bats":
        matdata = io.loadmat(args.data_dir+'/GC_bats/dataset_bats.mat')
        valid_samples = []
        # test_samples = [0,1]
        # len_samples = 1
        args.max = 1 
        num_seqs = len(matdata["dataset"][0])
        args.Fs = 30.3030

    else: # if args.experiment == 'sula' or args.experiment == 'peregrine' or args.experiment == 'mice' or args.experiment == 'flies':
        # args.data_dir = './datasets/animals/'
        data_ = np.load(os.path.join(args.data_dir,'GC_'+args.experiment+'/'+args.experiment+'_data.npy'))
        num_seqs = len(data_) # no. of sequences

        if args.experiment == 'peregrine':
            n_dim = 3
            args.Fs = 5
            valid_samples = []
            # test_samples = [0]#,1]
        elif args.experiment == 'sula':
            n_dim = 2
            args.Fs = 1
            valid_samples = []
            # test_samples = range(25)
        elif args.experiment == 'mice':
            n_dim = 2  
            args.Fs = 30
            valid_samples = []
            # test_samples = [0,1,2]
        elif args.experiment == 'flies':
            n_dim = 2  
            args.Fs = 30
            valid_samples = []
            # test_samples = [0,1]
        elif args.experiment == 'zebrafish':
            n_dim = 2  
            args.Fs = 30
            valid_samples = []
        args.max = 1 
        data = [[] for _ in range(num_seqs)]
        for i in range(num_seqs):
            if args.experiment == 'peregrine':
                data[i] = data_[i].transpose(2,1,0).astype(np.float64)
            else:
                data_i = np.array(data_[i])
                data[i] = np.zeros((len(data_i[0][0]),data_i.shape[1],data_i.shape[0]))
                for j in range(data_i.shape[1]): 
                    for k in range(data_i.shape[0]): 
                        data[i][:,j,k] = data_i[k,j].astype(np.float64)

    # time_length, num_dims, num_agants = data[0].shape
    args.test_samples = len(test_samples)
    args_original = copy.deepcopy(args)

    for f in range(num_seqs):
        del args
        args = copy.deepcopy(args_original)
        
        if args.experiment == "bats":
            loc = matdata["dataset"][0,f][0,0][0]
            vel = matdata["dataset"][0,f][0,0][1]
            index_none = matdata["dataset"][0,f][0,0][2].squeeze()   
            label = matdata["dataset"][0,f][0,0][3].squeeze()   
            
        else: # if args.experiment == 'sula' or args.experiment == 'peregrine' or args.experiment == 'mice' or args.experiment == 'flies':
            index_none = None
            vel = data[f][:,:n_dim]
            loc = data[f][:,n_dim:n_dim*2]
            label = None

        args.T, args.out_dims, args.p = loc.shape
        args.num_atoms = args.p 
        dataset, _, args, _ = animal_data(args,loc,vel,index_none=index_none,file_no=f)

        args.batch_size = args.T-args.K
        
        edges = None       
        '''if args.experiment == "bats":
            edges= np.zeros((args.p,args.p))
            for i in range(args.p):
                for j in range(args.p):
                    if i != j and np.sum(np.abs(label[:,i,j]))>0:
                        edges[i,j] = 1
        elif args.experiment == 'peregrine' or args.experiment == 'sula' or args.experiment == 'mice':
            edges = np.ones((args.p,args.p))
            np.fill_diagonal(edges, 0)
            if args.experiment == 'peregrine':
                edges[1,0] = -1'''

        if f in valid_samples:
            datasets_val.append(dataset) 
            index_None_val.append(index_none)
            args_list_val.append(copy.deepcopy(args))
            structures_val.append(edges)
            dynamic_structures_val.append(label)
            
        elif f in test_samples:
            datasets_te.append(dataset) 
            index_None_te.append(index_none)
            args_list_te.append(copy.deepcopy(args))
            structures_te.append(edges)
            dynamic_structures_te.append(label)

    signed_structures = structures_te if args.TEST else structures_val
    

    # hyperparameters
    if args.self_other and args.CF_pred: 
        lambdas = np.array([1.])
        betas = np.array([0.])
        gammas = np.array([1000.])
    else:
        lambdas = np.array([10.]) 
        betas = np.array([0.])
        gammas = np.array([0.]) 

elif "boid" in args.experiment:
    args.data_dir = './datasets/boid'   
    print("p = 5, T = 200 ")
    args.p = 5
    args.T = 200
    args.Fs = 1
    
    suffix = "_" + args.experiment + "_partial_avoid_l" + str(args.T) + "_Fs100" 
        
    args.out_dims = 2
    args.suffix = suffix

    args.num_atoms = args.p 

    args.max = 30 #np.max(feat[:,:,:,0])
    args.min = -30 # np.min(feat[:,:,:,0])

    datasets_val, structures_val, _, dynamic_structures_val = boid_data(args,suffix,train=0)
    datasets_te, structures_te, _, dynamic_structures_te = boid_data(args,suffix,train=-1)

    signed_structures = structures_te if args.TEST else structures_val

    # hyperparameters
    if args.TEST:
        if args.self_other: 
            lambdas = np.array([0.01])
            betas = np.array([0.025])
            if args.percept:
                lambdas = np.array([0.01])
                betas = np.array([0.])
            if args.CF_pred:
                lambdas = np.array([0.01])
                betas = np.array([0.025])
                gammas = np.array([1000.])
                if args.percept:
                    lambdas = np.array([1.])
                    betas = np.array([0.])
                    gammas = np.array([1000.])
        else:
            lambdas = np.array([1.])
            betas = np.array([0.025])
            gammas = np.array([0.])
    else:
        lambdas = np.logspace(-2, 0, 3, base=10)
        betas = np.linspace(0, 0.025, 2)
        gammas = np.logspace(1, 4, 4, base=10)

elif args.experiment == "kuramoto":
    datasets = []
    structures = []
    args.data_dir = './datasets/kuramoto'   
    print("p = 5, T = 100 ")
    args.p = 5
    args.T = 200
    args.Fs = 40

    suffix = '_kuramoto64'
    num_dims = 2
    out_dims = 1 
    args.d_other = 1

    args.num_dims = num_dims
    args.d_self = args.num_dims
    args.out_dims = out_dims
    args.suffix = suffix

    args.num_atoms = args.p 
    pred_len = args.T

    half_distance = 1e-6
    sigma_kernel = (half_distance**2)/(2*np.log(2)) # 0.5 when normalized distance is half_distance
    args.sigma_kernel = sigma_kernel

    # max_min
    suffix = 'valid' + suffix if not args.TEST else 'test' + suffix
    feat = np.load(os.path.join(args.data_dir, 'feat_' + suffix + ".npy"))
    edges = np.load(os.path.join(args.data_dir, 'edges_' + suffix + ".npy"))
    args.max = np.max(feat[:,:,:,0])
    args.min = np.min(feat[:,:,:,0])

    n, num_atoms, num_timesteps, all_dims = feat.shape
    # phase_diff, sin_theta, cos_theta, init_phase_diff, intrinsic_freq, phase
    
    feat = feat[:,:,:pred_len,:].transpose((0,2,1,3)) # self.num_dims
    # normalize
    feat[:,:,:,0] = feat[:,:,:,0] / args.max
    feat[:,:,:,4] = feat[:,:,:,4] / args.max
    feat[:,:,:,5] = (feat[:,:,:,5] + np.pi) % (2 * np.pi) - np.pi # phase

    for i in range(args.test_samples*2):
        feat_i = feat[i].copy()    
        edge = edges[i].copy()    
        if args.self_other:
            K = num_atoms
            for k in range(K):
                idx = np.arange(K)
                idx = np.delete(idx,k)
                sinjcosk = feat_i[:,idx,1] * feat_i[:,k,2:3].repeat(K-1,1)
                cosjsink = feat_i[:,idx,2] * feat_i[:,k,1:2].repeat(K-1,1)
                if args.d_other == 2:
                    loc_ = np.expand_dims(np.concatenate((sinjcosk,cosjsink),1),1)
                elif args.d_other == 1: 
                    loc_ = np.expand_dims(cosjsink - sinjcosk,1) # i - j
                vec = np.concatenate([vec,loc_],1) if k > 0 else loc_ 

            feat_final = np.concatenate((feat_i[:,:,0:1],feat_i[:,:,4:5],vec),2)
        else: 
            feat_final = np.concatenate((feat_i[:,:,0:1],feat_i[:,:,4:5]),2)

        num_dims = feat_final.shape[2]
        feat_final = feat_final.reshape((pred_len-1, feat_final.shape[2]*num_atoms)) # (p,dim)->dim*p
        if not args.TEST: # i < args.test_samples: 
            datasets_val.append(feat_final)
            structures_val.append(edge)
        else:
            datasets_te.append(feat_final)
            structures_te.append(edge)
    
    args.num_dims = num_dims

    # hyperparameters
    if args.TEST:
        if args.self_other: 
            lambdas = np.array([0])
            betas = np.array([0])
            
            if args.CF_pred:
                lambdas = np.array([0.])
                betas = np.array([0.])
                gammas = np.array([10.])
        else:
            lambdas = np.array([0.05])
            betas = np.array([0.025])

    else:
        lambdas = np.linspace(0, 0.1, 3)
        betas = np.linspace(0, 0.025, 2)
        gammas = np.logspace(-1, 2, 4, base=10)

else:
    NotImplementedError("ERROR: This experiment is not supported!")

if not args.CF_pred:
    gammas = np.array([0.0])

print('lambdas: '+ str(lambdas))
print('betas: '+ str(betas))
print('gammas: '+ str(gammas))

if not args.realdata: # args.experiment == "lorenz96":
    args.batch_size = args.T-args.K

#else:
#    args.batch_size = args.T-args.K-1

# Perform inference
# GVAR model
if args.model == "gvar":
    print("Model:           GVAR...")

    if args.TEST: 
        run_grid_search(args=args, lambdas=lambdas, gammas=gammas, betas=betas, datasets=datasets_te, K=args.K, structures=structures_te,
                        num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size,
                        num_epochs=args.num_epochs, batch_size=args.batch_size, initial_lr=args.initial_lr,
                        beta_1=args.beta_1, beta_2=args.beta_2, seed=args.seed,
                        signed_structures=signed_structures, dynamic_structures = dynamic_structures_te, 
                        index_None = index_None_te, args_list = args_list_te)
    else:
        run_grid_search(args=args, lambdas=lambdas, gammas=gammas, betas=betas, datasets=datasets_val, K=args.K, structures=structures_val,
                        num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size,
                        num_epochs=args.num_epochs, batch_size=args.batch_size, initial_lr=args.initial_lr,
                        beta_1=args.beta_1, beta_2=args.beta_2, seed=args.seed,
                        signed_structures=signed_structures, dynamic_structures = dynamic_structures_val,
                        index_None = index_None_val, args_list = args_list_val)

else:
    NotImplementedError("ERROR: Model is not supported!")