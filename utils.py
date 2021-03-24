# Some basic helper functions
import numpy as np
import warnings
import torch

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score


def construct_training_dataset(data, order, args, index_none=None):
    # Pack the data, if it is not in a list already
    if not isinstance(data, list):
        data = [data]
    
    data_out = None
    response = None
    time_idx = None
    # Iterate through time series replicates
    offset = 0

    for r in range(len(data)):
        data_out, response, time_idx = construct_dataset(data, r, order, offset, args, data_out, response, time_idx, index_none=index_none)

    return data_out, response, time_idx

def construct_dataset(data, r, order, offset, args, data_out, response, time_idx, index_none=None):
    data_r = data[r]

    # data: T x p
    T_r = data_r.shape[0]
    p_r = data_r.shape[1]
    inds_r = np.arange(order, T_r)
    K = args.num_atoms
    Din = args.num_dims
    Dout = args.out_dims
    data_out_r = np.zeros((T_r - order, order, p_r))
    response_r = np.zeros((T_r - order, K*Dout))
    time_idx_r = np.zeros((T_r - order, ))
    inds_res = []
    index0 = np.array(range(K*Din)).astype(int)
    for k in range(K):
        for d in range(Dout):
            inds_res = np.append(inds_res,index0[k*Din+d])
    inds_res = inds_res.astype(int)
    for i in range(T_r - order):
        j = inds_r[i] # output after the shift of the order 
        try: 
            data_out_r[i, :, :] = data_r[(j - order):j, :]
            response_r[i] = data_r[j, inds_res]
            if index_none is not None:
                k_idx = np.arange(K)
                idx_none = k_idx[np.where(np.all(index_none[(j - order):j,:]==1,axis=0))]
                
                for d in range(Dout):
                    idx_none_res = np.concatenate([idx_none_res,idx_none*Dout+d]) if d > 0 else idx_none*Dout
                # idx_none_res = inds_res[idx_none_res]
                response_r[i,idx_none_res] = np.ones((len(idx_none_res)))*9999
        except: import pdb; pdb.set_trace()
        time_idx_r[i] = j
    # TODO: just a hack, need a better solution...
    time_idx_r = time_idx_r + offset + 200 * (r >= 1)
    time_idx_r = time_idx_r.astype(int)

    if data_out is None:
        data_out = data_out_r
        response = response_r
        time_idx = time_idx_r
    else:
        data_out = np.concatenate((data_out, data_out_r), axis=0)
        response = np.concatenate((response, response_r), axis=0)
        time_idx = np.concatenate((time_idx, time_idx_r))
    offset = np.max(time_idx_r)
    return data_out, response, time_idx


def eval_causal_structure(a_true: np.ndarray, a_pred: np.ndarray, diagonal=True, max_fpr=0.5):
    # warnings.simplefilter('error')
    # if warning, often all zero (not a error)
    if diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))]
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))]
        if np.max(a_true_offdiag) == np.min(a_true_offdiag):
            auroc = None
            auprc = None
        else:
            auroc = roc_auc_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
            auprc = average_precision_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
            pauc = roc_auc_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten(),max_fpr=max_fpr)
    else:
        auroc = roc_auc_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
        auprc = average_precision_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
        pauc = roc_auc_score(y_true=a_true.flatten(), y_score=a_pred.flatten(),max_fpr=max_fpr)
    return auroc, auprc, pauc

def eval_causal_structure_binary(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        precision = precision_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        recall = recall_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        accuracy = accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        bal_accuracy = balanced_accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
    else:
        try:
            precision = precision_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
            recall = recall_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
            accuracy = accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
            bal_accuracy = balanced_accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        except: import pdb; pdb.set_trace()
    return accuracy, bal_accuracy, precision, recall
