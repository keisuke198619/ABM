# Training procedures for GVAR
import random, math, os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import MSELoss
from torch.autograd import Variable

import numpy as np

from utils import construct_training_dataset
from utils import eval_causal_structure, eval_causal_structure_binary
from models.senn import SENNGC

from sklearn.metrics import balanced_accuracy_score

import matplotlib.pyplot as plt


# Keisuke Fujii, 2021
# modifying the work and code: 
#   Interpretable Models for Granger Causality Using Self-explaining Neural Networks 
#   Ričards Marcinkevičs, Julia E Volgae, https://openreview.net/forum?id=DEa4JdMWRHp

def run_epoch_each(epoch_num, model, optimizer, predictors, responses, seqNo, time_idx, structure, criterion, lmbd, beta, gamma, batch_size, device, 
                i, index_none, inds, batch_split, incurred_loss, alpha, verbose, train, args, CF_pred, backward):
    
    catdim = 1
    p = args.num_atoms
    d_out = args.out_dims
    d_in = args.num_dims
    inds_ = []
    index0 = np.array(range(p*d_in)).astype(int)
    for k in range(p):
        for d in range(d_out):
            inds_ = np.append(inds_,index0[k*d_in+d])
    inds_ = torch.tensor(inds_, dtype=torch.long)

    if i < len(batch_split) - 1:
        predictors_b = predictors[inds[batch_split[i]:batch_split[i + 1]], :, :]
        responses_b = responses[inds[batch_split[i]:batch_split[i + 1]], :]
        time_idx_b = time_idx[inds[batch_split[i]:batch_split[i + 1]]]
    else:
        predictors_b = predictors[inds[batch_split[i]:], :, :]
        responses_b = responses[inds[batch_split[i]:], :]
        time_idx_b = time_idx[inds[batch_split[i]:]]

    inputs = Variable(torch.tensor(predictors_b, dtype=torch.float)).float().to(device)
    targets = Variable(torch.tensor(responses_b, dtype=torch.float)).float().to(device)
    
    # Get the forecasts and generalized coefficients
    preds, coeffs, coeffs_percept = model(inputs=inputs,CF_pred=False,inds=inds_,index_none=index_none)
    
    if args.self_other and "boid" in args.experiment: # (or args.experiment == "mices"):
        preds, _ = _clamp(preds, inputs, args, backward, inds_)

    preds = preds.reshape(preds.shape[0],args.num_atoms*args.out_dims)

    if not CF_pred:
        coeffs_CF = []
    # Loss
    # Base loss
    order,Dout,Din = coeffs.shape[1:]
    if index_none is not None:
        NoneIdx = (targets>9998).nonzero()
        NotNones = (targets<9998).nonzero()
        base_loss = criterion(preds[NotNones[:,0],NotNones[:,1]], targets[NotNones[:,0],NotNones[:,1]]) # 
        coeffs[NoneIdx[:,0],:,NoneIdx[:,1],:] = torch.zeros((NoneIdx.shape[0],order,Din)).to(device) # torch.log(-1*)
    else:
        base_loss = criterion(preds, targets)

    if args.percept:
        d_out = args.out_dims 
        d_self = args.d_self-d_out
    else:
        coeffs_percept = []
    # Sparsity-inducing penalty term
    # coeffs.shape:     [seq x T x K x p x p]
    coeffs_ = coeffs.clone()
    penalty = torch.zeros(1).to(device)
    d_self = args.d_self
    if args.self_other:
        seqs = coeffs.shape[0]
        p = args.num_atoms
        d_out = args.out_dims 
        d_in = args.num_dims
        d_in_coeff = coeffs.shape[3]

        # group penalty
        for k in range(p): # output
            idx_t = (targets[:,k*d_out]<9998).nonzero().squeeze()
            idx_gout = torch.arange(d_out*p)
            idx_gout = idx_gout[k*d_out:(k+1)*d_out]
            for j in range(p-1): # input
                if args.navigation or "kuramoto" in args.experiment:      
                    idx_gin = torch.arange(d_in_coeff)
                    if args.navigation:
                        d_self = args.d_self-d_out
                        idx_gin = idx_gin[d_self+j*d_out:d_self+(j+1)*d_out]
                    coeffs__ = coeffs[idx_t].clone(); coeffs__ = coeffs__[:,:,idx_gout]
                    if args.navigation: 
                        coeffs__ = coeffs__[:,:,:,idx_gin]

                penalty = computePenalty(penalty,args,coeffs__,alpha,p)

    else: 
        # coeffs.shape:     [T x K x (p x out) x (p x in)]
        penalty = (1 - alpha) * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=2), dim=0)) + \
                alpha * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=1), dim=0))

    # Smoothing penalty term
    penalty_smooth = torch.zeros(1).to(device)
    next_time_points = time_idx_b + 1
    inputs_next = Variable(torch.tensor(predictors[np.where(np.isin(time_idx, next_time_points))[0], :, :],
                                        dtype=torch.float)).float().to(device)
    preds_next, coeffs_next, _ = model(inputs=inputs_next,inds=inds_,index_none=index_none)
    
    if index_none is not None:
        for k in range(p):
            idx_t = (targets[:,k*d_out]<9998).cpu().numpy()
            idx_gout = torch.arange(d_out*p)
            idx_gout = idx_gout[k*d_out:(k+1)*d_out]
            coeffs__ = coeffs[idx_t&np.isin(next_time_points, time_idx)].clone(); coeffs__ = coeffs__[:,:,idx_gout]
            coeffs_next__ = coeffs_next[idx_t[1:]]; coeffs_next__ = coeffs_next__[:,:,idx_gout]
            if coeffs__.shape[0] > coeffs_next__.shape[0]:
                coeffs__ = coeffs__[:-1]
            elif coeffs__.shape[0] < coeffs_next__.shape[0]:
                coeffs_next__ = coeffs_next__[:-1]
            penalty_smooth += torch.mean(torch.norm(coeffs_next__ - coeffs__, dim=(2,3), p=2), dim=(0,1))/p
    else:
        penalty_smooth = torch.mean(torch.norm(coeffs_next - coeffs[np.isin(next_time_points, time_idx), :, :, :], dim=(2,3), p=2), dim=(0,1))
    
    CF_loss = torch.sqrt(torch.zeros(1).to(device))
    
    if CF_pred:
        for k in range(p):
            coeffs_list = [[] for _ in range(p-1)]
            idx_t = (targets[:,k*d_out]<9998).nonzero().squeeze()
            inputs_ = inputs[idx_t,-1,k*d_in:k*d_in+d_out] 
            targets_= targets[idx_t,k*d_out:(k+1)*d_out] 
            dist = torch.norm(inputs_-targets_,dim=1,p=2)
            gausskernel = torch.exp(-(dist)**2/(2*args.sigma_kernel))
            coeffs_ = coeffs[idx_t].clone(); coeffs_ = coeffs_[:,:,k*d_out:(k+1)*d_out]
            if not "kuramoto" in args.experiment: 
                coeffs_ = coeffs_[:,:,:,d_self:] 
            coeffs_zero = coeffs_*gausskernel.repeat((order,1,d_out*(p-1),1)).permute((3,0,1,2))

            if args.navigation:
                CF_loss += torch.mean(torch.norm(torch.max(torch.abs(coeffs_zero), dim=1)[0], dim=(1,2), p=2)) 
            elif "kuramoto" in args.experiment:
                CF_loss += torch.mean(torch.mean(torch.norm(coeffs_zero, dim=(2,3), p=2), dim=1)) 

        coeffs_CF = None
    
    # loss
    loss = base_loss + lmbd * penalty + beta * penalty_smooth + gamma * CF_loss

    # Incur loss
    incurred_loss[0] += loss.data.cpu().numpy()
    incurred_loss[1] += base_loss.data.cpu().numpy() # reconstruction
    incurred_loss[2] += penalty.data.cpu().numpy() # regularization lmbd * 
    incurred_loss[3] += penalty_smooth.data.cpu().numpy() # smoothness gamma * 
    incurred_loss[4] += CF_loss.data.cpu().numpy() # theory-guided (counterfactual)

    if train:
        # Make an optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #if epoch_num == 100:
    #    import pdb; pdb.set_trace()
    #    from scipy.io import savemat; mdic = {"coeffs_percept": coeffs_percept[:,0].detach().cpu().numpy()} ; savemat("coeffs_percept.mat", mdic)
    return coeffs, incurred_loss, coeffs_CF, coeffs_percept, preds

def computePenalty(penalty,args,coeffs,alpha,p):
    if args.navigation or "kuramoto" in args.experiment:
        dim1 = (2,3)
        dim2 = 1

    if args.navigation:
        penalty += (1 - alpha) * torch.mean(torch.norm(torch.max(torch.abs(coeffs), dim=1)[0], dim=(1,2), p=2)) + \
                                    alpha * torch.mean(torch.norm(torch.max(torch.abs(coeffs), dim=1)[0], dim=(1,2), p=1))
    elif "kuramoto" in args.experiment:
        penalty += (1 - alpha) * torch.mean(torch.mean(torch.norm(coeffs, dim=dim1, p=2), dim=dim2)) + \
                                alpha * torch.mean(torch.mean(torch.norm(coeffs, dim=dim1, p=1), dim=dim2)) # /p/(p-1)
        
    return penalty

def run_epoch(epoch_num: int, model: nn.Module, optimizer: optim, predictors: np.ndarray, responses: np.ndarray,
              seqNo: int, index_none: np.ndarray, time_idx: np.ndarray, structure: np.ndarray, criterion: torch.nn.modules.loss, lmbd: float, beta: float, gamma: float, batch_size: int,
              device: torch.device, alpha=0.5, verbose=True, train=True, args=None, CF_pred=False, backward=False):
    """
    Runs one epoch through the dataset.

    @param epoch_num: number of the epoch (for bookkeeping only).
    @param model: model.
    @param optimizer: Torch optimizer.
    @param predictors: numpy array with predictor values of shape [N x K x p].
    @param responses: numpy array with response values of shape [N x p].
    @param time_idx: time indices of observations of shape [N].
    @param criterion: base loss criterion (e.g. MSE or CE).
    @param lmbd: weight of the sparsity-inducing penalty.
    @param gamma: weight of the time-smoothing penalty.
    @param batch_size: batch size.
    @param device: Torch device.
    @param alpha: alpha-parameter for the elastic-net (default: 0.5).
    @param verbose: print-outs enabled?
    @param train: training mode?
    @return: if train == False, returns generalized coefficient matrices and average losses incurred; otherwize, None.
    """

    num_timesteps, order, num_dims = predictors.shape
    num_var = args.num_atoms
    p = args.num_atoms

    # Shuffle the data
    inds = np.arange(0, num_timesteps)
    if False: # train:
        np.random.shuffle(inds)

    if args.self_other:
        alpha = 0.5 # elastic net
    else:
        alpha = 0.1 # group sparse lasso

    # Split into batches
    batch_split = np.arange(0, len(inds), batch_size) # need to correct
    #if len(inds) - batch_split[-1] < batch_size / 2:
    #    batch_split = batch_split[:-1]
    
    if num_timesteps-batch_split[-1]==1:
        print('error: num_timesteps-batch_split[-1] should not 1 ')
        import pdb; pdb.set_trace()

    incurred_loss = np.zeros(5)

    for i in range(len(batch_split)):
        coeffs, incurred_loss, coeffs_CF,coeffs_percept,preds = run_epoch_each(
            epoch_num, model, optimizer, predictors, responses, seqNo, time_idx, structure, criterion, lmbd, beta, gamma, batch_size, device, 
            i, index_none, inds, batch_split, incurred_loss,
            alpha, verbose, train, args, CF_pred, backward)
    
    if epoch_num%100 == 0: # verbose:==1500: # 
        causal_struct_estimate, dynamic_causal_struct = estimate_causal_struct(coeffs,args,epoch_num) 
        causal_struct_estimate_ = dynamic_causal_struct if args.navigation and args.dynamic_edge else causal_struct_estimate

        if not args.realdata:
            for ii in range(p):
                if args.navigation and args.dynamic_edge:
                    if ii == 0:
                        structure_ = structure[args.K:,0:1,1:]
                    else:
                        structure__ = np.concatenate([structure[args.K:,ii:ii+1,:ii],structure[args.K:,ii:ii+1,ii+1:]],2)
                        structure_ = np.concatenate([structure_,structure__],1)  
                else:
                    structure_ = np.concatenate([structure_,structure[ii,:ii],structure[ii,ii+1:]],0) if ii > 0 else structure[0,1:]                
            if args.navigation: # "boid" in args.experiment:
                structure_ = np.abs(np.sign(structure_)).astype(np.int)

            auroc_l, auprc_l, pauc_l = eval_causal_structure(a_true=np.abs(structure_).ravel(), a_pred=np.abs(causal_struct_estimate_).ravel(), diagonal=False, max_fpr=0.1)
            print("Epoch " + str(epoch_num) + " : base_loss " + str(np.round(incurred_loss[1],4)) + "; sparsity " +
                str(np.round(incurred_loss[2],4)) + "; smoothness " + str(np.round(incurred_loss[3],4))+ "; CF_pred " + str(np.round(incurred_loss[4],4)) + 
                "; AUROC: " + str(np.round(auroc_l, 4))+ "; AUPRC: " + str(np.round(auprc_l, 4)) + "; pAUC: " + str(np.round(pauc_l, 4))  )
        else:
            print("Epoch " + str(epoch_num) + " : base_loss " + str(np.round(incurred_loss[1],4)) + "; sparsity " +
                str(np.round(incurred_loss[2],4)) + "; smoothness " + str(np.round(incurred_loss[3],4))+ "; CF_pred " + str(np.round(incurred_loss[4],4)))

    if not train:
        return coeffs, causal_struct_estimate, dynamic_causal_struct, coeffs_percept, preds, incurred_loss[0] / len(batch_split), incurred_loss[1] / len(batch_split), \
               incurred_loss[2] / len(batch_split), incurred_loss[3] / len(batch_split)

def _clamp(preds, inputs, args, backward, inds=None):
    """
    :param loc: 2xN location at one time stamp
    :param vel: 2xN velocity at one time stamp
    :return: location and velocity after hitting walls and returning after
        elastically colliding with walls
    """

    Fs = 1
    v0 = inputs[:,-1,inds] # batch,order,dim
    p0 = inputs[:,-1,inds+2]
    vel = preds # .reshape(preds.shape[0],args.num_atoms,args.out_dims)

    v0 = v0 * args.max
    p0 = p0 * args.max
    

    if not backward:
        loc = p0 + v0/Fs
    else: 
        loc = p0 - v0/Fs

    over = loc > args.max
    
    loc[over] = 2 * args.max - loc[over]
    assert torch.all(loc <= args.max)
    

    vel[over] = -torch.abs(vel[over])

    under = loc < -args.max
    loc[under] = -2 * args.max - loc[under]

    assert torch.all(loc >= -args.max)
    vel[under] = torch.abs(vel[under])

    return vel, loc

def estimate_causal_struct(coeffs,args,epoch_num):
    coeffs = coeffs.unsqueeze(0)
    seqs, T, order, d_out, d_in = coeffs.shape
    p = args.num_atoms
    d_self = args.d_self
    
    if args.navigation or args.experiment == 'kuramoto':
        # feat: vel*2,loc*2,[dir*2,dist]*(p-1)
        arange_in = torch.arange(d_in)
        arange_out = torch.arange(d_out)
        
        if args.self_other:
            for j in range(p-1): # in
                if not 'kuramoto' in args.experiment: # args.reduction:
                    if args.navigation:
                        d_self = args.d_self - d_out//p
                        inds_others = arange_in[d_self+j*d_out//p:d_self+(j+1)*d_out//p]
                    if j == 0:
                        coeffs_ = coeffs[:,:,:,:,inds_others].unsqueeze(5).clone()
                    else: 
                        coeffs_ = torch.cat([coeffs_,coeffs[:,:,:,:,inds_others].unsqueeze(5)],5) 
                else:
                    coeffs_ = coeffs.unsqueeze(5)
            d_in = (d_in - d_self)//(p-1) if not 'kuramoto' in args.experiment else d_in//(p-1)
        else: 
            for k in range(p): # out
                inds_out = arange_out[d_out//p*k:d_out//p*(k+1)]
                jj = 0
                for j in range(p): # in
                    if k != j:
                        inds_in = arange_in[jj*d_in//p:(jj+1)*d_in//p]
                        if jj == 0:
                            if len(inds_out)>1: # args.navigation:
                                coeffs___ = coeffs[:,:,:,inds_out].clone()
                                coeffs__ = coeffs___[:,:,:,:,inds_in].unsqueeze(5)
                            else:
                                coeffs__ = coeffs[:,:,:,inds_out,inds_in].unsqueeze(3).unsqueeze(5)
                        else: 
                            if len(inds_out)>1: # args.navigation:
                                coeffs___ = coeffs[:,:,:,inds_out].clone()
                                coeffs__ = torch.cat([coeffs__,coeffs___[:,:,:,:,inds_in].unsqueeze(5)],5)  
                            else:
                                coeffs__ = torch.cat([coeffs__,coeffs[:,:,:,inds_out,inds_in].unsqueeze(3).unsqueeze(5)],5)  
                        jj += 1 
                
                coeffs_ = coeffs__ if k == 0 else torch.cat([coeffs_,coeffs__],3) 

            d_in = d_in//p

    # reshape
    # e.g., aa = torch.arange(60); aa.reshape((5,3,4)) 
    coeffs_ = coeffs_.reshape((T,order,p,d_out//p,d_in,p-1)).permute(0,1,3,4,2,5) # ..., p, p-1
    max_than_median = True

    if args.navigation: # and args.dynamic_edge:
        # causal_struct_estimate = torch.mean(torch.abs(coeffs_), dim=1) # about order
        if max_than_median:
            coeffs_ = coeffs_.cpu().detach().numpy()
            causal_struct_estimate = np.zeros((T,d_out//p,d_in,p,p-1))
            # mesh = np.meshgrid(np.arange(p),np.arange(T),np.arange(p-1))
            mesh = np.meshgrid(np.arange(p),np.arange(T))
            for din in range(d_in):
                for dout in range(d_out//p):
                    for pp in range(p-1):
                        try:
                            max_idx = np.argmax(np.abs(coeffs_[:,:,dout,din,:,pp]),1)
                            causal_struct_estimate[:,dout,din,:,pp] = coeffs_[mesh[1],max_idx,dout,din,mesh[0],pp]
                        except: import pdb; pdb.set_trace()
            dynamic_causal_struct = np.linalg.norm(causal_struct_estimate, axis=(1,2), ord=2)*np.sign(np.median(causal_struct_estimate.reshape((T,d_out//p*d_in,p,p-1)),axis=1))

            mesh = np.meshgrid(np.arange(p-1),np.arange(p))
            causal_struct_estimate = dynamic_causal_struct[np.argmax(np.abs(dynamic_causal_struct),0),mesh[1],mesh[0]]
        else: # median
            causal_struct_estimate = torch.median(coeffs_, dim=1)[0] # about order
            dynamic_causal_struct = torch.norm(causal_struct_estimate, dim=(1,2), p=2)*torch.sign(torch.median(causal_struct_estimate.reshape((T,d_out//p*d_in,p,p-1)),dim=1)[0]) # about d_out and d_in 
            causal_struct_estimate = torch.median(dynamic_causal_struct,dim=0)[0]
    elif args.experiment == 'kuramoto':
        causal_struct_estimate = torch.max(torch.abs(coeffs_), dim=1)[0] # about order
        dynamic_causal_struct = torch.mean(causal_struct_estimate, dim=(1,2)) # about d_out and d_in     
        causal_struct_estimate = torch.median(dynamic_causal_struct,dim=0)[0]   

    if args.navigation: 
        return causal_struct_estimate, dynamic_causal_struct
    else:
        return causal_struct_estimate.cpu().detach().numpy(), dynamic_causal_struct.cpu().detach().numpy()

def training_procedure(data, structure, args, order: int, hidden_layer_size: int, end_epoch: int, batch_size: int, lmbd: float, beta: float,
                       gamma: float, seqNo=0, index_none = None, num_hidden_layers=1, initial_learning_rate=0.001, beta_1=0.9, beta_2=0.999, use_cuda=True, 
                       verbose=True, test_data=None, CF_pred=False, backward=False, model=None):
    """
    Standard training procedure for GVAR model.

    @param data: numpy array with time series of shape [T x p].
    @param order: GVAR model order.
    @param hidden_layer_size: number of units in a hidden layer.
    @param end_epoch: number of training epochs.
    @param batch_size: batch size.
    @param lmbd: weight of the sparsity-inducing penalty.
    @param gamma: weight of the time-smoothing penalty.
    @param seed: random generator seed.
    @param num_hidden_layers: number oh hidden layers.
    @param initial_learning_rate: learning rate.
    @param use_cuda: whether to use GPU?
    @param verbose: print-outs enabled?
    @param test_data: optional test data.
    @return: returns an estimate of the GC dependency structure, generalized coefficient matrices, and the test MSE,
    if test data provided.
    """

     # Check for CUDA availability
    if use_cuda and not torch.cuda.is_available():
        print("WARNING: CUDA is not available!")
        device = torch.device("cpu")
    elif use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Number of variables, p
    num_vars = args.num_atoms # data[0].shape[1]
    predictors, responses, time_idx = construct_training_dataset(data=data, order=order, args=args, index_none=index_none)

    # Model definition
    if model is None:
        model = SENNGC(num_vars=num_vars, order=order, hidden_layer_size=hidden_layer_size,
                        num_hidden_layers=num_hidden_layers, device=device,args=args)
        model.to(device=device)

    # Loss criterion
    criterion = MSELoss()

    optimizer = optim.Adam(params=model.parameters(), lr=initial_learning_rate, betas=(beta_1, beta_2))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.995 ** epoch) # 1 
    # Run the training and testing
    for epoch in range(end_epoch):
        # Train
        run_epoch(epoch_num=epoch, model=model, optimizer=optimizer, predictors=predictors, responses=responses,
                  seqNo = seqNo, index_none=index_none, time_idx=time_idx, structure=structure, criterion=criterion, lmbd=lmbd, beta=beta, gamma=gamma, batch_size=batch_size,
                  device=device, train=True, verbose=verbose, args=args, CF_pred=CF_pred, backward=backward)
        
        #if epoch%100==0:
        #    print('epoch:{}, lr:{}'.format(epoch, scheduler.get_last_lr()[0])) 

        scheduler.step()


    # Compute generalized coefficients & estimate causal structure
    with torch.no_grad():
        coeffs, causal_struct_estimate, dynamic_causal_struct, coeffs_percept, preds, l, mse, pen1, pen2 = run_epoch(epoch_num=end_epoch, model=model, optimizer=optimizer,
                                               predictors=predictors, responses=responses, seqNo = seqNo, index_none= index_none, time_idx=time_idx, structure=structure,
                                               criterion=criterion, lmbd=lmbd, beta=beta, gamma=gamma, batch_size=batch_size, device=device, train=False, 
                                               verbose=verbose, args=args, CF_pred=CF_pred, backward=backward)
        coeffs_percept = coeffs_percept.cpu().numpy() if args.percept else None

        return causal_struct_estimate, dynamic_causal_struct, coeffs.cpu().numpy(), coeffs_percept, mse, model, preds.cpu().numpy()

def training_procedure_trgc(data, structure, args, order: int, hidden_layer_size: int, end_epoch: int, batch_size: int, lmbd: float, beta: float, 
                            gamma: float, seqNo=0, index_none=None, num_hidden_layers=1, initial_learning_rate=0.001, beta_1=0.9,
                            beta_2=0.999, Q=20, use_cuda=True, verbose=True, display=False, true_struct=None,
                            signed=False, bidirection=True, model=None):
    """
    Stability-based estimation of the GC structure using GVAR model and time reversed GC (TRGC). Sparsity level is
    chosen to maximize the agreement between GC structures inferred on original and time-reversed time series.

    @param data: numpy array with time series of shape [T x p].
    @param order: GVAR model order.
    @param hidden_layer_size: number of units in a hidden layer.
    @param end_epoch: number of training epochs.
    @param batch_size: batch size.
    @param lmbd: weight of the sparsity-inducing penalty.
    @param gamma: weight of the time-smoothing penalty.
    @param seed: random generator seed.
    @param num_hidden_layers: number oh hidden layers.
    @param initial_learning_rate: learning rate.
    @param Q: number of quantiles (spaced equally) to consider for thresholding (default: 20).
    @param use_cuda:  whether to use GPU?
    @param verbose: print-outs enabled?
    @param display: plot stability across considered sparsity levels?
    @param true_struct: ground truth GC structure (for plotting stability only).
    @param signed: detect signs of GC interactions?
    @return: an estimate of the GC summary graph adjacency matrix, strengths of GC interactions, and generalized
    coefficient matrices. If signed == True, in addition, signs of GC interactions are returned.
    """
    data_1 = None
    data_2 = None
    flipaxis = 0
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
        data_1 = data
        data_2 = np.flip(data, axis=flipaxis)
    else:
        data_1 = data
        data_2 = np.flip(data, axis=flipaxis)

    flag_dynamic = (args.navigation and args.dynamic_edge)

    if verbose:
        print("-" * 25)
        print("Running TRGC selection...")
        print("Training model #1...")

    a_hat_1_, a_hat_d1_, coeffs_full_1, coeffs_percept_1, rmse_1, model, preds = training_procedure(data=[data_1], structure=structure, args=args, order=order, hidden_layer_size=hidden_layer_size,
                                                end_epoch=end_epoch, lmbd=lmbd, beta=beta, gamma=gamma, batch_size=batch_size,
                                                seqNo=seqNo, index_none = index_none, num_hidden_layers=num_hidden_layers,
                                                initial_learning_rate=initial_learning_rate, beta_1=beta_1, beta_2=beta_2, use_cuda=use_cuda, 
                                                verbose=False, CF_pred=args.CF_pred, backward=False, model=model)
    if flag_dynamic:
        T = a_hat_d1_.shape[0]
        p = a_hat_1_.shape[-2]
        a_hat_1 = a_hat_d1_.copy()
        a_hat_1_original = a_hat_d1_.copy()
    else:
        p = a_hat_1_.shape[-2]
        T = 1 
        a_hat_1 = a_hat_1_.copy()
        a_hat_1_original = a_hat_1.copy()

    if bidirection:
        if verbose:
            print("Training model #2...")
        a_hat_2_, a_hat_d2_, coeffs_full_2, coeffs_percept_2, rmse_2, model_2, _ = training_procedure(data=[data_2], structure=structure, args=args, order=order, hidden_layer_size=hidden_layer_size,
                                                end_epoch=end_epoch, lmbd=lmbd, beta=beta, gamma=gamma, batch_size=batch_size,
                                                seqNo=seqNo, index_none = index_none, num_hidden_layers=num_hidden_layers,
                                                initial_learning_rate=initial_learning_rate, beta_1=beta_1, beta_2=beta_2, use_cuda=use_cuda, 
                                                verbose=False, CF_pred=args.CF_pred, backward=True, model=None)
        if flag_dynamic:
            a_hat_2 = a_hat_d2_.copy()
        else:
            a_hat_2 = a_hat_2_.copy()
            
        # add diagonal elements
        a_hat_2_diag = np.zeros((p,p,T))
        for k in range(p):
            jj = 0
            for j in range(p):
                if k != j:
                    if flag_dynamic:
                        a_hat_2_diag[k,j,:] = a_hat_2[:,k,jj]
                    else:
                        a_hat_2_diag[k,j,0] = a_hat_2[k,jj]
                    jj += 1  

        # transpose
        # see [Winkler+18, IEEE TSP] Validity of time reversal for testing Granger causality
        a_hat_2_diag = np.transpose(a_hat_2_diag,(1,0,2)) 
        
        # remove diagonal elements
        for k in range(p):
            jj = 0
            for j in range(p):
                if k != j:
                    if flag_dynamic:
                        a_hat_2[:,k,jj] = a_hat_2_diag[k,j,:]
                    else:
                        a_hat_2[k,jj] = a_hat_2_diag[k,j,0] 
                    jj += 1

        # normalize
        
        a_hat_1 = np.abs(a_hat_1/np.max(np.abs(a_hat_1)))
        a_hat_2 = np.abs(a_hat_2/np.max(np.abs(a_hat_2)))    

        if verbose:
            print("Evaluating stability...")
        alphas = np.linspace(0, 1, Q)
        qs_1 = np.quantile(a=a_hat_1, q=alphas)
        qs_2 = np.quantile(a=a_hat_2, q=alphas)
        if flag_dynamic:
            agreements = np.zeros((len(alphas), T))
            agreements_ground = np.zeros((len(alphas), T)) if true_struct is not None else None
        else:
            agreements = np.zeros((len(alphas), ))
            agreements_ground = np.zeros((len(alphas), )) if true_struct is not None else None
        
        for i in range(len(alphas)):
            a_1_i = (a_hat_1 >= qs_1[i]) * 1.0
            a_2_i = (a_hat_2 >= qs_2[i]) * 1.0
            if not flag_dynamic:
                agreements[i] = (balanced_accuracy_score(y_true=a_2_i.flatten(),y_pred=a_1_i.flatten()) +
                                balanced_accuracy_score(y_pred=a_2_i.flatten(),y_true=a_1_i.flatten())) / 2

                # If only self-causal relationships are inferred, then set agreement to 0
                if np.sum(a_1_i) == 0 or np.sum(a_2_i) == 0:
                    agreements[i] = 0
                
                # If all potential relationships are inferred, then set agreement to 0
                if np.sum(a_1_i) == p*(p-1) or np.sum(a_2_i) == p*(p-1):
                    agreements[i] = 0

                if true_struct is not None:
                    agreements_ground[i] = balanced_accuracy_score(y_true=true_struct.flatten(),
                                                                y_pred=a_1_i.flatten())
            else:
                for t in range(T):
                    agreements[i,t] = (balanced_accuracy_score(y_true=a_2_i[t].flatten(),y_pred=a_1_i[t].flatten()) +
                                balanced_accuracy_score(y_pred=a_2_i[t].flatten(),y_true=a_1_i[t].flatten())) / 2
                    # If only self-causal relationships are inferred, then set agreement to 0
                    if np.sum(a_1_i[t]) == 0 or np.sum(a_2_i[t]) == 0:
                        agreements[i,t] = 0
                    
                    # If all potential relationships are inferred, then set agreement to 0
                    if np.sum(a_1_i[t]) == p*(p-1) or np.sum(a_2_i[t]) == p*(p-1):
                        agreements[i,t] = 0

                    if true_struct is not None:
                        agreements_ground[i,t] = balanced_accuracy_score(y_true=true_struct[t].flatten(),
                                                                    y_pred=a_1_i[t].flatten())
        if not flag_dynamic:
            alpha_opt = alphas[np.argmax(agreements)]
            if True: # verbose:
                print("Max. stab. = " + str(np.round(np.max(agreements), 3)) + ", at α = " + str(alpha_opt))

            q_1 = np.quantile(a=a_hat_1, q=alpha_opt)
            q_2 = np.quantile(a=a_hat_2, q=alpha_opt)
            a_hat_binary = (a_hat_1 >= q_1) * 1.0
            a_signed = a_hat_1_original * a_hat_binary
        else:
            alpha_opt = np.zeros((T,))
            a_hat_binary = np.zeros((T,p,p-1))
            a_signed = np.zeros((T,p,p-1))
            for t in range(T):
                alpha_opt[t] = alphas[np.argmax(agreements[:,t])]
                q_1 = np.quantile(a=a_hat_1[t], q=alpha_opt[t])
                q_2 = np.quantile(a=a_hat_2[t], q=alpha_opt[t])
                a_hat_binary[t] = (a_hat_1[t] >= q_1) * 1.0
                a_signed[t] = a_hat_1_original[t] * a_hat_binary[t]

            if False: #  display: # TBD 
                plot_stability(alphas, agreements, agreements_ground=agreements_ground)
            if True: # verbose:
                max_agreements_m = np.round(np.mean(np.max(agreements,axis=0)),3)
                max_agreements_sd = np.round(np.std(np.max(agreements,axis=0)),3)
                alpha_opt_m = np.round(np.mean(alpha_opt),3)
                alpha_opt_sd = np.round(np.std(alpha_opt),3)               
                print("Max. stab. = " + str(max_agreements_m) + " (+/-) " + str(max_agreements_sd) 
                    + ", at α = " + str(alpha_opt_m)+ " (+/-) " + str(alpha_opt_sd))

        rmse = rmse_1 # (rmse_1+rmse_2)/2
    else:
        
        if not flag_dynamic:
            a_hat_1_original = a_hat_1.copy()
            a_hat_binary = np.abs(a_hat_1).copy()
            
            if args.navigation:
                val_max_pos = np.max(a_hat_1_original)
                val_max_neg = -np.min(a_hat_1_original)
                threshold_pos = val_max_pos/2
                threshold_neg = val_max_neg/2
                a_signed = a_hat_1.copy()
                for i in range(p):
                    a_signed[i,a_signed[i]>=threshold_pos] = 1
                    a_signed[i,a_signed[i]<=-threshold_neg] = -1
                    a_signed[i,(a_signed[i]<threshold_pos)&(a_signed[i]>-threshold_neg)] = 0
                a_hat_binary = np.abs(a_signed)

            elif args.experiment == 'kuramoto':
                val_max = np.max(a_hat_1_original)
                threshold = val_max/2 
                a_hat_binary[a_hat_binary>=threshold] = 1
                a_hat_binary[a_hat_binary<threshold] = 0
                a_signed = a_hat_1_original * a_hat_binary

        elif flag_dynamic: # TBD
            import pdb; pdb.set_trace()
        else:
            a_hat_binary = a_hat_1
            a_signed = a_hat_1
        rmse = rmse_1
        
    if not signed:
        if flag_dynamic:
            return a_hat_binary, a_hat_1_, a_hat_1, coeffs_full_1, coeffs_percept_1, rmse, model, preds
        else:
            return a_hat_binary, a_hat_1, a_hat_d1_, coeffs_full_1, coeffs_percept_1, rmse, model, preds
    else:
        if flag_dynamic:
            return a_hat_binary, a_hat_1_original, a_hat_1_, a_signed, coeffs_full_1, coeffs_percept_1, rmse, model, preds
        else:
            return a_hat_binary, a_hat_1_original, a_signed, a_hat_d1_, coeffs_full_1, coeffs_percept_1, rmse, model, preds
