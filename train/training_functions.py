import torch
import numpy as np

def channel_confusion(x_a, x_b):
    swap_inds = [1, 0, 3, 2, 5, 4]

    if x_a.shape[1]==6:
        for j in range(6):
            if np.random.randint(2) == 1:
                x_a[j, :, :, :] = x_a[j:j+1, swap_inds, :, :]
            if np.random.randint(2) == 1:
                x_b[j, :, :, :] = x_b[j:j+1, swap_inds, :, :]

    return x_a, x_b

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    x_a, x_b = x, x[index, :]
    x_a, x_b = channel_confusion(x_a, x_b)
    mixed_x = lam * x_a + (1 - lam) * x_b
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
