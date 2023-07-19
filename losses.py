# Third party import
import logging
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances, cosine_similarity, laplacian_kernel
import numpy as np


class SupConLoss(nn.Module):
    def __init__(self,
                 config,
                 temperature: float = 0.1,
                 return_logits: bool = False):
        """
        Supervised Contrastive Learning loss. For more details, refer to
        Prannay Khosla, Piotr Teterwak, Chen Wang et al.
        Supervised Contrastive Learning, NeurIPS 2020.
        :param config: configuration file containing the main info for training
        :param temperature: 'tau' parameter specific to InfoNCE loss
        :param return_logits: boolean parameter if return the correct pairs in the similarity matrix.
        :return: a PyTorch Module.
        """

        super().__init__()
        self.temperature = temperature
        self.config = config
        self.return_logits = return_logits

    def forward(self, z_i, z_j, labels):
        N = len(z_i)
        id_mat = torch.eye(2*N, device=z_i.device)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]
        z = torch.cat([z_i,z_j], dim=0)         # dim [2N, D]
        sim_mat = (z @ z.T) / self.temperature  # shape [2N, 2N]
        sim_mat = ( sim_mat * ( 1 - id_mat ) ) - id_mat * 1e8  # similarity matrix: the diag has to be removed
        labs = func.one_hot(torch.tensor(labels, device=z_i.device).long(),num_classes=self.config.num_classes)
        L = torch.cat([labs,labs], dim=0).to(torch.float32)       # shape [2N, 2]: one-hot encoded vector of labels, repeated twice
        mask = L @ L.T                          # shape [2N, 2N]: mask where mask(i,j) = 1 if z_i(i) has same label as z_j(j) and 0 otherwise
        mask = mask * (1 - id_mat)
        card_P_i = mask.sum(dim=1)

        log_sim = func.log_softmax(sim_mat,dim=1)
        positive_mat = (log_sim * mask) / card_P_i
        loss = -positive_mat.sum() / (2*N)

        correct_pairs = torch.arange(N, device=z_i.device).long()
        sim_zij = sim_mat[:N,N:]                # the upper right matrix contains z_i @ z_j.T

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss




class WSPContrastiveLoss(nn.Module):
    def __init__(self,
                 config,
                 kernel: str = 'rbf',
                 temperature: float = 0.1,
                 return_logits: float = False,
                 sigma: float = 1.0):
        """
        The proposed WSP contrastive loss. For more details, refer to:
        Emma Sarfati, Alexandre Bône, Marc-Michel Rohé, Pietro Gori, Isabelle Bloch
        Weakly-supervised positional contrastive learning: application to cirrhosis classification, MICCAI 2023.
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma' which corresponds to gamma=1/(2*sigma**2)
        :param temperature: the tau parameter of the NTXentLoss.
        :param return_logits: boolean parameter if return the correct pairs in the similarity matrix.
        :param sigma: the sigma value of the Gaussian kernel.
        :return: a PyTorch Module.
        """

        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        self.config = config
        if self.kernel == 'rbf':
            self.kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        if self.kernel == 'laplacian':
            self.kernel = lambda y1, y2: laplacian_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        elif self.kernel == 'discrete':
            self.kernel = lambda y1, y2: vec_weights(euclidean_distances(y1,y2))
        elif self.kernel == 'distance':
            self.kernel = lambda y1, y2: vec(euclidean_distances(y1,y2))
        else:
            assert hasattr(self.kernel, '__call__'), 'kernel must be a callable'
        self.temperature = temperature
        self.return_logits = return_logits
        self.INF = 1e8

    def forward(self, z_i, z_j, labels, z_pos):
        N = len(z_i)
        num_classes = self.config.num_classes
        assert N == len(labels), "Unexpected labels length: %i"%len(labels)
        z_i = func.normalize(z_i, p=2, dim=-1)     # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)     # dim [N, D]
        sim_zii = (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)


        # binary mask computed with variable y in the paper
        labs = func.one_hot(torch.tensor(labels, device=z_i.device).long(), num_classes=num_classes)
        L = torch.cat([labs, labs], dim=0).to(
            torch.float32)                  # shape [2N, 2]: one-hot encoded vector of labels, repeated twice
        mask = L @ L.T                      # shape [2N, 2N]: mask where mask(i,j) = 1 if z_i(i) has same label as z_j(j) and 0 otherwise
        mask = mask * (1 - torch.eye(2*N)).to(z_i.device)    # puts 0 on the diagonal

        # continuous weights mask computed with variable d in the paper
        all_labels = torch.tensor(z_pos).view(N, -1).repeat(2, 1).detach().cpu().numpy()  # [2N, *]
        weights = self.kernel(all_labels, all_labels)                                     # [2N, 2N]
        weights = torch.from_numpy(weights * (1 - np.eye(2*N))).to(z_i.device)            # puts 0 on the diagonal

        # final array of mask, dot product between dirac and exponential + normalization
        final_weights = weights * mask
        final_weights /= final_weights.sum(dim=1)

        # compute the loss
        sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1), torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0) # [2N, 2N]
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = -1./N * (log_sim_Z * final_weights).sum()

        correct_pairs = torch.arange(N, device=z_i.device).long()

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss




class NTXenLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning.
    For more details, refer to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations, ICML 2020.
    :param temperature: the 'tau' parameter referred in the paper.
    :param return_logits: boolean parameter if return the correct pairs in the similarity matrix.
    :return: PyTorch module.
    """

    def __init__(self,
                 temperature: float = 0.1,
                 return_logits: bool = False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii = (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)

        if self.return_logits:
            return loss_i + loss_j, sim_zij, correct_pairs

        return loss_i + loss_j

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)
