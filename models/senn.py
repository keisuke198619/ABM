import torch.nn as nn
import torch
import math

# Keisuke Fujii, 2021
# modifying the work and code: 
#   Interpretable Models for Granger Causality Using Self-explaining Neural Networks 
#   Ričards Marcinkevičs, Julia E Volgae, https://openreview.net/forum?id=DEa4JdMWRHp

class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device,
                 args=None):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.

        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        @param method: fitting algorithm (currently, only "OLS" is supported).
        """
        super(SENNGC, self).__init__()

        # Networks for amortising generalised coefficient matrices.
        
        self.num_atoms = args.num_atoms
        self.self_other = args.self_other
        self.percept = args.percept
        self.num_dims = args.num_dims 
        out_dims = args.out_dims
        self.experiment = args.experiment
        self.realdata = args.realdata
        self.navigation = args.navigation
        
        self.d_self = args.d_self
        self.num_vars = num_vars
        if self.navigation:
            if self.self_other: 
                self.num_dims += - (num_vars - 1)*2

        elif 'kuramoto' in self.experiment:
            self.d_other = args.d_other
            self.max = args.max

        
        # Instantiate coefficient networks
        self.coeff_nets = nn.ModuleList()
        if self.self_other: #  and not self.permuted:
            if self.navigation:
                num_dims = self.num_dims-out_dims# (self.num_dims-self.d_self)#//(num_vars - 1)
                # out_dims = 1
                num_dims2 = num_dims# (num_vars-1)
            elif 'kuramoto' in self.experiment:
                num_dims = num_vars # -1
                num_dims2 = num_dims -1
            else:
                num_dims = self.num_dims-out_dims # if not args.percept else self.num_dims - (num_vars-1)
                num_dims2 = num_dims
            for k in range(order):
                self.coeff_nets_ = nn.ModuleList()
                for i in range(num_vars):
                    modules = [nn.Sequential(nn.Linear(num_dims, hidden_layer_size), nn.ReLU())]
                    if num_hidden_layers > 1:
                        for j in range(num_hidden_layers - 1):
                            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
                    if 'kuramoto' in self.experiment:
                        modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_dims2*out_dims,nn.Sigmoid()))) # 
                    else:
                        modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_dims2*out_dims)))     
                    self.coeff_nets_.append(nn.Sequential(*modules))
                self.coeff_nets.append(self.coeff_nets_)
        else:
            num_dims = self.num_dims #  if not args.percept else self.num_dims - 1
            for k in range(order):
                modules = [nn.Sequential(nn.Linear(num_vars*num_dims, hidden_layer_size), nn.ReLU())]
                if num_hidden_layers > 1:
                    for j in range(num_hidden_layers - 1):
                        modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
                    if 'kuramoto' in self.experiment:
                        modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars*num_dims*num_vars*out_dims))) 
                    else:
                        modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars*num_dims*num_vars*out_dims)))     
                self.coeff_nets.append(nn.Sequential(*modules))

        if args.percept:
            self.avoid_nets = nn.Linear(1,1)
            self.avoid_nets.apply(self.init_weights_avoid)

        # Some bookkeeping
        
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers

        self.device = device
        self.out_dims = out_dims
        self.out_dims_final = args.out_dims
        self.num_dims_model = num_dims

    # Initialization
    def init_weights_avoid(self,m):
        if 'bat' in self.experiment:
            nn.init.constant_(m.bias, 0.1)
        else:
            nn.init.constant_(m.bias, 0.)


    # Forward propagation,
    # returns predictions and generalized coefficients corresponding to each prediction
    def forward(self, inputs: torch.Tensor, CF_pred=False, inds=None, index_none=None):
        coeffs = None

        if index_none is not None:
            index_none = torch.tensor(-(index_none-1)).bool().to(self.device)

        timelength, order, input_dim = inputs.shape
        input_dim //= self.num_vars
        preds = torch.zeros((timelength, self.num_vars*self.out_dims_final)).to(self.device)

        p = self.num_vars
        d_out = self.out_dims
        dim = self.out_dims_final

        if self.percept:
            indds = torch.zeros(0).long()
            coeffs_FP = torch.zeros((timelength,p, p-1)).to(self.device)
            avoid = self.avoid_nets.bias 
            for i in range(p):
                for j in range(p-1):
                    indd = (i+1)*input_dim - (p-1) + j 
                    inputs_FP = inputs[:, -1, indd].clone()
                    indd2 = (i+1)*input_dim - (p-1)*2 + j 
                    inputs_FP2 = inputs[:, -1, indd2].clone()
                    if index_none is None:
                        if 'bat' in self.experiment:
                            coeffs_FP[:,i,j] = torch.sigmoid(1e2*(inputs_FP - avoid)) * (torch.sigmoid(1e2*inputs_FP2)-1/2)*2
                        else:
                            coeffs_FP[:,i,j] = torch.sigmoid(1e6*(inputs_FP )) * (torch.sigmoid(1e2*inputs_FP2)-1/2)*2
                    else:
                        if index_none.shape[0]-inputs.shape[1]-1 == timelength:
                            tmp_idx = index_none[inputs.shape[1]:-1,i]# .shape
                        else: 
                            tmp_idx = index_none[inputs.shape[1]:,i]
                    
                        coeffs_FP[tmp_idx,i,j] = torch.sigmoid(1e2*(inputs_FP[tmp_idx] - avoid)) * (torch.sigmoid(1e2*inputs_FP2[tmp_idx])-1/2)*2

        else: 
            coeffs_FP = None
        
        for k in range(self.order):
            coeff_net_k = self.coeff_nets[k]

            if self.self_other:
                for i in range(p):
                    if self.navigation or 'kuramoto' in self.experiment:
                        if self.navigation:
                            idx = torch.arange(input_dim*i + self.num_dims)
                            idx = torch.cat([idx[input_dim*i:input_dim*i+dim],idx[input_dim*i+2*dim:input_dim*i+(2+p-1)*dim]],dim=0)
                            inputs_ = inputs[:, k, idx].clone()
                        elif 'kuramoto' in self.experiment:
                            inputs_ = torch.cat([inputs[:, k, input_dim*i:input_dim*i+1]-inputs[:, k, input_dim*i+4:input_dim*i+5],inputs[:, k, input_dim*i+2:input_dim*i+self.num_dims]],dim=1)
                        else:
                            inputs_ = inputs[:, k, input_dim*i:input_dim*i + self.num_dims].clone()
                    elif 'lorenz' in self.experiment: # delayed processing: same as not self_other???
                        inputs_ = inputs[:, k, :].clone()
                    coeffs_ki = coeff_net_k[i](inputs_)
                    
                    if self.navigation:  
                        if self.percept:
                            coeffs_ki = torch.reshape(coeffs_ki**2, (timelength, self.out_dims, self.num_dims_model))
                        else: 
                            coeffs_ki = torch.reshape(coeffs_ki, (timelength, self.out_dims, self.num_dims_model))
                    else:
                        if 'kuramoto' in self.experiment:
                            coeffs_ki = torch.reshape(coeffs_ki**2, (timelength, p-1, self.out_dims))
                        else: 
                            coeffs_ki = torch.reshape(coeffs_ki, (timelength, p-1, self.out_dims))

                    coeffs_ki2 = coeffs_ki.clone()
                    if self.percept:
                        coeffs_ki2[:,:,-(p-1)*dim:] = coeffs_ki[:,:,-(p-1)*dim:] * coeffs_FP[:,i,:].unsqueeze(2).unsqueeze(2).expand(timelength, p-1, self.out_dims, self.out_dims).reshape((timelength,(p-1)*self.out_dims,self.out_dims)).permute((0,2,1))

                    if 'kuramoto' in self.experiment:
                        inputs__ = inputs[:, k, input_dim*i:input_dim*i + self.num_dims].clone()
                    else:
                        inputs__ = inputs_.clone()

                    if 'kuramoto' in self.experiment:
                        preds__ = inputs__[:,1:2]/self.order + torch.sum(coeffs_ki2*inputs__[:,2:].unsqueeze(dim=2),dim=1)
                        coeffs_ki2 = coeffs_ki2.permute((0,2,1))
                    else:
                        preds__ = torch.matmul(coeffs_ki2, inputs__.unsqueeze(dim=2)).squeeze(2)

                    preds_ = torch.cat([preds_,preds__],1) if i > 0 else preds__     
                    coeffs_k = torch.cat([coeffs_k,coeffs_ki2],1) if i > 0 else coeffs_ki2
                    if torch.sum(torch.isnan(preds_))>0:
                        import pdb; pdb.set_trace()
                preds += preds_
                
            else:
                coeffs_k = coeff_net_k(inputs[:, k, :])
                coeffs_k = torch.reshape(coeffs_k, (timelength, self.num_vars*self.out_dims, self.num_vars*self.num_dims_model))
                preds += torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze(2)

            coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1) if coeffs is not None else torch.unsqueeze(coeffs_k, 1)

        return preds, coeffs, coeffs_FP
