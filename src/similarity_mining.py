"""Similarity modules and loss modules for LoRS
by Yue Xu"""

from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F



class BaseSimilarityGenerator(nn.Module, ABC):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def generate_with_param(self, params: List):
        pass

    @abstractmethod
    def get_indexed_parameters(self, indices=None) -> List:
        pass

    def load_params(self, params):
        raise NotImplementedError("")
    
    

class LowRankSimilarityGenerator(BaseSimilarityGenerator):
    """Generate a parameterized similarity matrix S = aI + L@R' 
    where a is a vector, I is the identity matrix, and B is a matrix of learnable parameters."""
    def __init__(self, dim, rank, alpha=0.1):
        super().__init__(dim)
        self.rank = float(rank)

        self.alpha = alpha

        self.diag_weight = nn.Parameter(torch.ones(dim))
        self.left = nn.Parameter(torch.randn(dim, rank))
        self.right = nn.Parameter(torch.zeros(dim, rank))

    
    def generate_with_param(self, params: List):
        a, l, r = params

        diag = torch.diag(a)
        sim = diag + l @ r.t() * self.alpha / self.rank
        return sim
    
    def get_indexed_parameters(self, indices=None) -> List:
        if indices is None:
            a, l, r = self.diag_weight, self.left, self.right
        else:
            a, l, r = self.diag_weight[indices], self.left[indices], self.right[indices]

        return [a, l, r]

    def load_params(self, params):
        a, l, r = params
        self.diag_weight.data = a
        self.left.data  = l
        self.right.data = r



class FullSimilarityGenerator(BaseSimilarityGenerator):
    def __init__(self, dim):
        super().__init__(dim)
        self.sim_mat = nn.Parameter(torch.eye(dim))
            
    def generate_with_param(self, params):
        assert len(params) == 1
        return params[0]

    def get_indexed_parameters(self, indices=None) -> List:
        if indices is None:
            sim = self.sim_mat
        else:
            sim = self.sim_mat[indices[:,None], indices]
        return [sim]
    
    def load_params(self, params):
        s = params[0]
        self.sim_mat.data = s





class MultilabelContrastiveLoss(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, loss_type=None):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super().__init__()
        self.loss_type = loss_type

        self.kl_loss_func = nn.KLDivLoss(reduction="mean")
        self.bce_loss_func = nn.BCELoss(reduction="none")

        self.n_clusters = 4   # for KmeansBalanceBCE


        
    def __kl_criterion(self, logit, label):
        # batchsize = logit.shape[0]
        probs1 = F.log_softmax(logit, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.kl_loss_func(probs1, probs2)
        return loss

    def __cwcl_criterion(self, logit, label):
        logprob = torch.log_softmax(logit, 1)
        loss_per = (label * logprob).sum(1) / (label.sum(1)+1e-6)
        loss = -loss_per.mean()
        return loss
    
    def __infonce_nonvonventional_criterion(self, logit, label):
        logprob = torch.log_softmax(logit, 1)
        loss_per = (label * logprob).sum(1)
        loss = -loss_per.mean()
        return loss
        

    def forward(self, logits, gt_matrix):
        gt_matrix = gt_matrix.to(logits.device)

        if self.loss_type == "KL":
            loss_i = self.__kl_criterion(logits, gt_matrix)
            loss_t = self.__kl_criterion(logits.t(), gt_matrix.t())
            return (loss_i + loss_t) * 0.5
        elif self.loss_type == "BCE":
            # print("BCE is called")
            probs1 = torch.sigmoid(logits)
            probs2 = gt_matrix # torch.sigmoid(gt_matrix)
            bce_loss = self.bce_loss_func(probs1, probs2)
            loss = bce_loss.mean()
            # loss = self.__general_cl_criterion(logits, gt_matrix, "BCE", 
                                            #    use_norm=False, use_negwgt=False)
            return loss
        
        elif self.loss_type in ["BalanceBCE", "WBCE"]:
            probs1 = torch.sigmoid(logits)
            probs2 = gt_matrix # torch.sigmoid(gt_matrix)

            loss_matrix = - probs2 * torch.log(probs1+1e-6) - (1-probs2) * torch.log(1-probs1+1e-6)
            
            pos_mask = (probs2>0.5).detach()
            neg_mask = ~pos_mask

            loss_pos = torch.where(pos_mask, loss_matrix, torch.tensor(0.0, device=probs1.device)).sum()
            loss_neg = torch.where(neg_mask, loss_matrix, torch.tensor(0.0, device=probs1.device)).sum()
            
            loss_pos /= (pos_mask.sum()+1e-6)
            loss_neg /= (neg_mask.sum()+1e-6)

            return (loss_pos+loss_neg)/2

        
        elif self.loss_type in ["NCE", "InfoNCE"]:
            loss_i = self.__infonce_nonvonventional_criterion(logits, gt_matrix)
            loss_t = self.__infonce_nonvonventional_criterion(logits.t(), gt_matrix.t())
            return (loss_i + loss_t) * 0.5

        elif self.loss_type == "MSE":
            probs = torch.sigmoid(logits)
            return F.mse_loss(probs, gt_matrix)
        
        elif self.loss_type == "CWCL":
            loss_i = self.__cwcl_criterion(logits, gt_matrix)
            loss_t = self.__cwcl_criterion(logits.t(), gt_matrix.t())
            return (loss_i + loss_t) * 0.5

        else:
            raise NotImplementedError(self.loss_type)