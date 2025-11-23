import torch

def get_nn_grad(net):
    param_grads = [];
    for param in net.parameters():
        param_grads.append(param.grad.reshape(-1))
    grads = torch.cat(param_grads);
    return grads
        

