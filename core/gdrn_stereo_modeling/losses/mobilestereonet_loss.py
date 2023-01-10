# authored by https://github.com/cogsys-tuebingen/mobilestereonet

import torch.nn.functional as F


###############################################################################
""" Loss Function """
###############################################################################

def msn_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est * mask, disp_gt * mask, reduction='mean'))
    return sum(all_losses)