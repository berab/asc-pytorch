import torch
import numpy as np

def channel_confusion(x_a, x_b, use_cuda=True):
    swap_inds = [1, 0, 3, 2, 5, 4]

    
    if x_a.shape[1]==6:
        print('channel conf')
        for j in torch.arange(6).cuda():
            if np.random.randint(2) == 1:
                x_a[j, :, :, :] = x_a[j:j+1, swap_inds, :, :]
            if np.random.randint(2) == 1:
                x_b[j, :, :, :] = x_b[j:j+1, swap_inds, :, :]

    return x_a, x_b

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if use_cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    batch_size = x.shape[0]
    if alpha > 0:
        dist = torch.distributions.beta.Beta(torch.zeros(batch_size)+alpha, torch.zeros(batch_size)+alpha)
    else:
        dist = torch.ones(batch_size).to(device)
    
    lam = dist.sample().to(device)
    X_l = lam.reshape(batch_size,1,1,1)
    y_l = lam.reshape(batch_size)
    
    index = torch.randperm(batch_size).to(device)
    #import pdb; pdb.set_trace() 
    x_a, x_b = x, x[index, :]
    y_a, y_b = y, y[index]
    x_a, x_b = channel_confusion(x_a, x_b, use_cuda)
    #import pdb; pdb.set_trace()
    mixed_x = X_l * x_a + (1 - X_l) * x_b
    import pdb; pdb.set_trace()
    mixed_y = y_a * y_l + y_b * (1 - y_l)
    
    return mixed_x, mixed_y

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
