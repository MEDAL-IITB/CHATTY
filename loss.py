import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

class Myloss(nn.Module):
    def __init__(self,epsilon=1e-8):
        super(Myloss,self).__init__()
        self.epsilon = epsilon
        return
    def forward(self,input_, label, weight):
        entropy = - label * torch.log(input_ + self.epsilon) -(1 - label) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy * weight)/2 
    
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ *torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

   
def MCC(input_vec):

    outputs_temp = input_vec / 0.5 
    softmax_out_temp = nn.Softmax(dim=1)(outputs_temp)
    entropy_weight = Entropy(softmax_out_temp).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    
    entropy_weight = input_vec.size(0) * entropy_weight / torch.sum(entropy_weight)
    cov_matrix = softmax_out_temp.mul(entropy_weight.view(-1,1)).transpose(1,0).mm(softmax_out_temp) #########
    cov_matrix = cov_matrix / torch.sum(cov_matrix, dim=1)
    mcc_loss = (torch.sum(cov_matrix) - torch.trace(cov_matrix)) / input_vec.size(1)
    
    return mcc_loss, cov_matrix
    

def DB(input_list, ad_net, coeff=None, myloss=Myloss()):

    torch.autograd.set_detect_anomaly(True)
    softmax_out_1, softmax_out_2 = input_list[0], input_list[1]

    #NEED TO CREATE A NEW SOFTMAX OUTPUT PROPORTIONAL TO softmax_out_1 * softmax_out_2
    ad_out,fc_out = ad_net((softmax_out_1+softmax_out_2)/2.0)
    ad_out = nn.Sigmoid()(ad_out - fc_out)
    batch_size = softmax_out_1.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()

    x1, x2 = softmax_out_1, softmax_out_2
    entropy1, entropy2 = Entropy(x1), Entropy(x2)

    entropy1.register_hook(grl_hook(coeff))
    entropy2.register_hook(grl_hook(coeff))

    entropy1, entropy2 = torch.exp(-entropy1), torch.exp(-entropy2)
    mean_entropy1,  mean_entropy2 = torch.mean(entropy1), torch.mean(entropy2)

    source_mask_1, source_mask_2 = torch.ones_like(entropy1.detach()), torch.ones_like(entropy2.detach())
    source_mask_1[softmax_out_1.size(0)//2:] = 0
    source_mask_2[softmax_out_2.size(0)//2:] = 0

    source_weight_1, source_weight_2= entropy1.detach()*source_mask_1, entropy2.detach()*source_mask_2

    target_mask_1, target_mask_2 = torch.ones_like(entropy1), torch.ones_like(entropy2)
    target_mask_1[softmax_out_1.size(0)//2:] = 0
    source_mask_2[softmax_out_2.size(0)//2:] = 0

    target_weight_1, target_weight_2= entropy1*target_mask_1, entropy2*target_mask_2

    weight1 = source_weight_1 / torch.sum(source_weight_1).detach().item() + target_weight_1 / torch.sum(target_weight_1).detach().item()
    weight2 = source_weight_2 / torch.sum(source_weight_2).detach().item() + target_weight_2 / torch.sum(target_weight_2).detach().item()
    
    return myloss(ad_out,dc_target,weight1.view(-1, 1)), myloss(ad_out,dc_target,weight2.view(-1, 1)), mean_entropy1, mean_entropy2	




