import os, random

import time

import numpy as np
import torch

from utils import eval_causal_structure, eval_causal_structure_binary

from training import training_procedure_trgc
from scipy.io import savemat

# Keisuke Fujii, 2021
# modifying the work and code: 
#   Interpretable Models for Granger Causality Using Self-explaining Neural Networks 
#   Ričards Marcinkevičs, Julia E Volgae, https://openreview.net/forum?id=DEa4JdMWRHp

def run_grid_search(args, lambdas: np.ndarray, gammas: np.ndarray, betas: np.ndarray, datasets: list, structures: list, K: int,
                    num_hidden_layers: int, hidden_layer_size: int, num_epochs: int, batch_size: int,
                    initial_lr: float, beta_1: float, beta_2: float, seed: int, 
                    signed_structures=None, dynamic_structures = None, index_None = None, args_list = None):
    """
    Evaluates GVAR model across a range of hyperparameters.

    @param lambdas: values for the sparsity-inducing penalty parameter.
    @param gammas: values for the smoothing penalty parameter.
    @param datasets: list of time series datasets.
    @param structures: ground truth GC structures.
    @param K: model order.
    @param num_hidden_layers: number of hidden layers.
    @param hidden_layer_size: number of units in a hidden layer.
    @param num_epochs: number of training epochs.
    @param batch_size: batch size.
    @param initial_lr: learning rate.
    @param beta_1 and beta_2: parameters of ADAM.
    @param seed: random generator seed.
    @param dynamic_structures: ground truth dynamic GC structures.
    @param index_None, index_None_val, index_None_te: None data indices.
    @param signed_structures: ground truth signs of GC interactions.
    """
    # Check for CUDA availability
    if args.use_cuda and not torch.cuda.is_available():
        print("WARNING: CUDA is not available!")
        device = torch.device("cpu")
    elif args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Set random generator seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # save weights
    wtdir = "./weights/" + args.experiment + "_" + args.model + "_" + str(args.test_samples) + "/"

    filename0 = ''
    if args.TEST:
        filename0 = filename0 + '_TEST'
    if args.percept:
        filename0 = filename0 + '_percept'
    if args.CF_pred:
        filename0 = filename0 + '_CF_pred'
    if args.self_other:
        filename0 = filename0 + '_self' 
    if args.bidirection:
        filename0 = filename0 + '_bidirection'

    filename0 = wtdir + filename0
    if not os.path.isdir(filename0):
        os.makedirs(filename0)

    # For binary structures
    use_threshold = True
    only_binary = False
    if not args.realdata:
        if use_threshold:
            mean_accs = np.zeros((len(lambdas), len(betas), len(gammas)))
            sd_accs = np.zeros((len(lambdas), len(betas), len(gammas)))
            mean_bal_accs = np.zeros((len(lambdas), len(betas), len(gammas)))
            sd_bal_accs = np.zeros((len(lambdas), len(betas), len(gammas)))
        else:
            mean_accs = None; sd_accs = None; mean_bal_accs = None; sd_bal_accs = None; 

        # For continuous structures
        mean_aurocs = np.zeros((len(lambdas), len(betas), len(gammas)))
        sd_aurocs = np.zeros((len(lambdas), len(betas), len(gammas)))
        mean_auprcs = np.zeros((len(lambdas), len(betas), len(gammas)))
        sd_auprcs = np.zeros((len(lambdas), len(betas), len(gammas)))
        mean_paucs = np.zeros((len(lambdas), len(betas), len(gammas)))
        sd_paucs= np.zeros((len(lambdas), len(betas), len(gammas)))

        # For effect signs
        if signed_structures is not None:
            mean_bal_accs_pos = np.zeros((len(lambdas), len(betas), len(gammas)))
            sd_bal_accs_pos = np.zeros((len(lambdas), len(betas), len(gammas)))
            mean_bal_accs_neg = np.zeros((len(lambdas), len(betas), len(gammas)))
            sd_bal_accs_neg = np.zeros((len(lambdas), len(betas), len(gammas)))
        else:
            mean_bal_accs_pos = None; sd_bal_accs_pos = None; mean_bal_accs_neg = None; sd_bal_accs_neg = None; 

    # for other indices
    mean_mse = np.zeros((len(lambdas), len(betas), len(gammas)))
    sd_mse = np.zeros((len(lambdas), len(betas), len(gammas)))
    mean_coeff = np.zeros((len(lambdas), len(betas), len(gammas)))
    sd_coeff= np.zeros((len(lambdas), len(betas), len(gammas)))

    flag_dynamic = (args.navigation and args.dynamic_edge)
    
    
    ijk = 0
    print("Iterating through " + str(len(lambdas)) + " x " + str(len(gammas)) + " grid of parameters...")
    for i in range(len(lambdas)):
        lmbd_i = lambdas[i]
        for j in range(len(gammas)):
            gamma_j = gammas[j]
            for k in range(len(betas)):
                beta_k = betas[k]
                print("λ = " + str(lambdas[i]) + "; β = " + str(beta_k)+ "; γ = " + str(gammas[j]) + "; progress: " +
                    str((ijk) / (len(gammas) * len(lambdas) * len(betas)) * 100) )
                filename = filename0 + '_lmb' + str(lambdas[i]) + '_bt' + str(betas[k]) + '_gm' + str(gammas[j])
                mse_ij = []
                coeff_ij = []
                if not args.realdata:
                    accs_ij = []
                    bal_accs_ij = []
                    prec_ij = []
                    rec_ij = []
                    aurocs_ij = []
                    auprcs_ij = []
                    paucs_ij = []
                    if signed_structures is not None:
                        bal_accs_pos_ij = []
                        bal_accs_neg_ij = []
                
                n_data = args.test_samples if args.TEST else args.test_samples 
                time_compute = np.zeros((n_data,1))
                for l in range(n_data): # 1,2):# 4,5):# 
                    if args.realdata:
                        args = args_list[l]
                        index_none = index_None[l]
                    else:
                        index_none = None
                    d_l = datasets[l]
                    a_l = structures[l]
                    a_dl = dynamic_structures[l] if dynamic_structures is not None else None

                    p = args.num_atoms
                    for ii in range(p):
                        if flag_dynamic:
                            if ii == 0:
                                a_l_ = a_l[args.K:,0:1,1:]
                            else:
                                a_l__ = np.concatenate([a_l[args.K:,ii:ii+1,:ii],a_l[args.K:,ii:ii+1,ii+1:]],2)
                                a_l_ = np.concatenate([a_l_,a_l__],1) 
                        elif not args.realdata:
                            a_l_ = np.concatenate([a_l_,a_l[ii,:ii],a_l[ii,ii+1:]],0) if ii > 0 else a_l[0,1:]
                        else:
                            a_l_ = None
                        if not flag_dynamic and a_dl is not None:
                            if ii == 0:
                                a_dl_ = a_dl[args.K:,0:1,1:]
                            else:
                                a_dl__ = np.concatenate([a_dl[args.K:,ii:ii+1,:ii],a_dl[args.K:,ii:ii+1,ii+1:]],2)
                                a_dl_ = np.concatenate([a_dl_,a_dl__],1) 
                        elif a_dl is None:
                            a_dl_ = []
                    if signed_structures is not None:
                        if args.realdata: 
                            a_l_ = a_l   
                        else:
                            a_l_signed = a_l_.copy()
                            a_l_ = np.abs(np.sign(a_l_)).astype(np.int)

                    batch_size = args.batch_size # d_l.shape[0] # 
                    start = time.time()

                    if signed_structures is None:
                        a_hat_l, a_hat_l_, a_hat_dl_, coeffs_full_l, coeffs_percept_l, mse, model, preds = training_procedure_trgc(data=d_l,structure=a_l, args=args, order=K,
                                                                                hidden_layer_size=hidden_layer_size,
                                                                                end_epoch=num_epochs, lmbd=lmbd_i, beta=beta_k,
                                                                                gamma=gamma_j, batch_size=batch_size,
                                                                                seqNo = l, index_none = index_none,
                                                                                num_hidden_layers=num_hidden_layers,
                                                                                initial_learning_rate=initial_lr,
                                                                                beta_1=beta_1, beta_2=beta_2, true_struct=a_l_,
                                                                                verbose=False, bidirection=args.bidirection)
                        a_hat_l_signed = []
                        a_l_signed = []
                    else:
                        # a_l_signed = signed_structures[l]
                        a_hat_l, a_hat_l_, a_hat_l_signed, a_hat_dl_signed, coeffs_full_l, coeffs_percept_l, mse,model, preds = training_procedure_trgc(data=d_l,structure=a_l,args=args, order=K,
                                                                                hidden_layer_size=hidden_layer_size,
                                                                                end_epoch=num_epochs, lmbd=lmbd_i, beta=beta_k, 
                                                                                gamma=gamma_j, batch_size=batch_size,
                                                                                seqNo = l, index_none = index_none,
                                                                                num_hidden_layers=num_hidden_layers,
                                                                                initial_learning_rate=initial_lr,
                                                                                beta_1=beta_1, beta_2=beta_2, true_struct=a_l_,
                                                                                verbose=False, signed=True, bidirection=args.bidirection)

                    time_compute[l] = time.time() - start 
                    mse = np.sqrt(mse)
                    mse_ij.append(mse)
                    coeff_ij.append(np.mean(np.abs(coeffs_full_l)))

                    if not args.realdata:
                        if only_binary:# 
                            if use_threshold: 
                                try: acc_l, bal_acc_l, prec_l, rec_l = eval_causal_structure_binary(a_true=np.abs(a_l_), a_pred=np.abs(a_hat_l).flatten(), diagonal=False)
                                except: import pdb; pdb.set_trace()
                                accs_ij.append(acc_l)
                                bal_accs_ij.append(bal_acc_l)
                                prec_ij.append(prec_l)
                                rec_ij.append(rec_l)

                                print("Dataset #" + str(l + 1) + "; Acc.: " + str(np.round(acc_l, 4)) + "; Bal. Acc.: " +
                                    str(np.round(bal_acc_l, 4)) + "; RMSE: " + str(np.round(mse, 4)) + 
                                    "; coeff.: " + str(np.round(np.mean(np.abs(coeffs_full_l)), 4)))#, end='\r'
                            else:
                                print("Dataset #" + str(l + 1) + "; RMSE: " + str(np.round(mse, 4)) + 
                                    "; coeff.: " + str(np.round(np.mean(np.abs(coeffs_full_l)), 4)))
                        else:
                            auroc_l, auprc_l, pauc_l = eval_causal_structure(a_true=np.abs(a_l_), a_pred=np.abs(a_hat_l_), diagonal=False, max_fpr=0.1) # p x p-1
                            aurocs_ij.append(auroc_l)
                            auprcs_ij.append(auprc_l)
                            paucs_ij.append(pauc_l)
                            
                            if use_threshold: 
                                acc_l, bal_acc_l, prec_l, rec_l = eval_causal_structure_binary(a_true=np.abs(a_l_), a_pred=np.abs(a_hat_l).flatten(), diagonal=False)
                                accs_ij.append(acc_l)
                                bal_accs_ij.append(bal_acc_l)
                                prec_ij.append(prec_l)
                                rec_ij.append(rec_l)

                                print("Dataset #" + str(l + 1) + "; Acc.: " + str(np.round(acc_l, 4)) + "; Bal. Acc.: " +
                                    str(np.round(bal_acc_l, 4)) #  +"; Prec.: " + str(np.round(prec_l, 4)) + "; Rec.: " + str(np.round(rec_l, 4)) 
                                    + "; AUROC: " + str(np.round(auroc_l, 4)) + "; AUPRC: " + str(np.round(auprc_l, 4)) +  
                                    "; pAUC: " + str(np.round(pauc_l, 4)) + "; RMSE: " + str(np.round(mse, 4)) + 
                                    "; coeff.: " + str(np.round(np.mean(np.abs(coeffs_full_l)), 4)))#, end='\r'
                            else:
                                print("Dataset #" + str(l + 1) + "; AUROC: " + str(np.round(auroc_l, 4)) + "; AUPRC: " + str(np.round(auprc_l, 4)) + 
                                    "; pAUC: " + str(np.round(pauc_l, 4)) + "; RMSE: " + str(np.round(mse, 4)) + 
                                    "; coeff.: " + str(np.round(np.mean(np.abs(coeffs_full_l)), 4)))#, end='\r' '''

                        if signed_structures is not None:
                            _, bal_acc_pos, __, ___ = eval_causal_structure_binary(a_true=(a_l_signed > 0).ravel() * 1.0,
                                                                                a_pred=(a_hat_l_signed > 0).ravel() * 1.0)
                            _, bal_acc_neg, __, ___ = eval_causal_structure_binary(a_true=(a_l_signed < 0).ravel() * 1.0,
                                                                                a_pred=(a_hat_l_signed < 0).ravel() * 1.0)
                            bal_accs_pos_ij.append(bal_acc_pos)
                            bal_accs_neg_ij.append(bal_acc_neg)
                            print("signed structures #" + str(l + 1) + "; Bal.Acc_pos: " + str(np.round(bal_acc_pos, 4)) + "; Bal.Acc_neg: " + str(np.round(bal_acc_neg, 4)) ) 
                        
                    if args.navigation and len(lambdas)==1: # and args.percept 
                        tmp_a_l = a_hat_l if signed_structures is None else a_hat_l_signed
                        if args.percept:
                            weights = model.avoid_nets.bias.data.detach().cpu().numpy()

                            if not args.realdata:
                                mdic = {"coeffs_raw": a_hat_l_,"coeffs": tmp_a_l,"data":d_l,"coeffs_time":a_hat_dl_signed,"weights":weights,"true_bi":a_l_,"true_time":a_dl_,"true_signed":a_l_signed,"args":args,"percept":coeffs_percept_l, "preds":preds}
                            else:
                                mdic = {"coeffs_raw": a_hat_l_,"coeffs": tmp_a_l,"data":d_l,"coeffs_time":a_hat_dl_signed,"weights":weights,"args":args,"percept":coeffs_percept_l, "preds":preds}

                        else:
                            if not args.realdata: 
                                mdic = {"coeffs_raw": a_hat_l_,"coeffs": tmp_a_l,"data":d_l,"coeffs_time":a_hat_dl_signed,"true_bi":a_l_,"true_time":a_dl_,"true_signed":a_l_signed,"args":args, "preds":preds}
                            else:
                                mdic = {"coeffs_raw": a_hat_l_,"coeffs": tmp_a_l,"data":d_l,"coeffs_time":a_hat_dl_signed,"args":args, "preds":preds}
                        savemat(filename0+"/coeffs_"+str(l+1)+".mat", mdic)
                        
                        if not args.realdata: 
                            print("predicted:" + str(a_hat_l_))    
                            print("ground truth:" + str(a_l_signed.reshape((p,p-1))))     
                        else:
                            print("Dataset #" + str(l + 1) + " analyzed")             
                        # np.round(a_hat_l_signed[:,0,0],1)
                    elif "kuramoto" in args.experiment and len(lambdas)==1:
                        coeffs_time = np.max(np.abs(coeffs_full_l[:,:,:,args.d_self:]), axis=1)
                        mdic = {"coeffs_bi": a_hat_l,"coeffs": a_hat_l_,"coeffs_time":coeffs_time,"data":d_l,"true_bi":a_l_,"args":args, "preds":preds}
                        savemat(filename0+"/coeffs_"+str(l+1)+".mat", mdic)   
                        print("predicted:" + str(a_hat_l_)) 
                        print("ground truth:" + str(a_l_.reshape((p,p-1))))     
                print()
                if not args.realdata: 
                    if use_threshold: 
                        mean_accs[i,k,j] = np.mean(accs_ij)
                        print("Acc.         :" + str(mean_accs[i,k,j]))
                        sd_accs[i,k,j] = np.std(accs_ij)
                        mean_bal_accs[i,k,j] = np.mean(bal_accs_ij)
                        print("Bal. Acc.    :" + str(mean_bal_accs[i,k,j]))
                        sd_bal_accs[i,k,j] = np.std(bal_accs_ij)

                    if not only_binary:
                        mean_aurocs[i,k,j] = np.mean(aurocs_ij)
                        print("AUROC        :" + str(mean_aurocs[i,k,j]))
                        sd_aurocs[i,k,j] = np.std(aurocs_ij)
                        mean_auprcs[i,k,j] = np.mean(auprcs_ij)
                        print("AUPRC        :" + str(mean_auprcs[i,k,j]))
                        sd_auprcs[i,k,j] = np.std(auprcs_ij)
                        mean_paucs[i,k,j] = np.mean(paucs_ij)
                        print("pAUC        :" + str(mean_paucs[i,k,j]))
                        sd_paucs[i,k,j] = np.std(paucs_ij)
                    
                    if signed_structures is not None:
                        mean_bal_accs_pos[i,k,j] = np.mean(bal_accs_pos_ij)
                        print("BA (pos.)    :" + str(mean_bal_accs_pos[i,k,j]))
                        sd_bal_accs_pos[i,k,j] = np.std(bal_accs_pos_ij)
                        mean_bal_accs_neg[i,k,j] = np.mean(bal_accs_neg_ij)
                        print("BA (neg.)    :" + str(mean_bal_accs_neg[i,k,j]))
                        sd_bal_accs_neg[i,k,j] = np.std(bal_accs_neg_ij)

                mean_mse[i,k,j] = np.mean(mse_ij)
                print("RMSE        :" + str(mean_mse[i,k,j]))
                sd_mse[i,k,j] = np.std(mse_ij)
                mean_coeff[i,k,j] = np.mean(coeff_ij)
                print("coeff.         :" + str(mean_coeff[i,k,j]))
                sd_coeff[i,k,j] = np.std(coeff_ij)

    if args.realdata: 
        i = 0 ; j = 0 ; k = 0 ; 
    elif use_threshold: 
        i,k,j = np.where(mean_bal_accs==np.max(mean_bal_accs))
    else:
        i,k,j = np.where(mean_auprcs==np.max(mean_auprcs))

    # display
    print(" computation time: {0:.0f} $\pm$ {1:.0f}".format(np.mean(time_compute), np.std(time_compute)))
    print(filename0 + " best λ = " + str(lambdas[i]) + "; β = " + str(betas[k])+ "; γ = " + str(gammas[j]))
    print(' RMSE, mean_coeffs: ' +  str(np.round(mean_mse[i,k,j],3))+' $\pm$ '+str(np.round(sd_mse[i,k,j],3))+' &'
        +' ' + str(np.round(mean_coeff[i,k,j],3))+' $\pm$ '+str(np.round(sd_coeff[i,k,j],3))+' ')
    if not args.realdata: 
        if signed_structures is not None:
            # BA, AUPRC, BA_pos, BA_neg
            print(   'BA, AUPRC, BA_pos, BA_neg:' 
                    +' ' +  str(np.round(mean_bal_accs[i,k,j],3))+' $\pm$ '+str(np.round(sd_bal_accs[i,k,j],3))+' &'
                    +' ' + str(np.round(mean_auprcs[i,k,j],3))+' $\pm$ '+str(np.round(sd_auprcs[i,k,j],3))+' &'
                    +' ' + str(np.round(mean_bal_accs_pos[i,k,j],3))+' $\pm$ '+str(np.round(sd_bal_accs_pos[i,k,j],3))+' &'
                    +' ' + str(np.round(mean_bal_accs_neg[i,k,j],3))+' $\pm$ '+str(np.round(sd_bal_accs_neg[i,k,j],3))+' ')   
        else:
            # Acc, BA, AUC, AUPRC 
            print(  'Acc, BA, AUC, AUPRC:'
                    +' ' + str(np.round(mean_accs[i,k,j],3))+' $\pm$ '+str(np.round(sd_accs[i,k,j],3))+' &'
                    +' ' + str(np.round(mean_bal_accs[i,k,j],3))+' $\pm$ '+str(np.round(sd_bal_accs[i,k,j],3))+' &'
                    +' ' + str(np.round(mean_aurocs[i,k,j],3))+' $\pm$ '+str(np.round(sd_aurocs[i,k,j],3))+' &'
                    +' ' + str(np.round(mean_auprcs[i,k,j],3))+' $\pm$ '+str(np.round(sd_auprcs[i,k,j],3))+' ')   
                    
    print('computation finished!')
    import pdb; pdb.set_trace()
