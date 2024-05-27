import torch
import torch.nn as nn
import torch.nn.functional as F

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)
    loss = F.kl_div(input=log_prob, target=one_hot, reduction='none', log_target=False)
    return loss.sum(-1)

def get_smooth_crossentropy(predictions, targets):
    return smooth_crossentropy(predictions, targets, smoothing=0.1).mean()