import matplotlib.pyplot as plt
import torch
device = 'cuda' if torch.cuda.is_available else 'cpu'
def make_one_hot(labels, C=3):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.LongTensor(labels.size(0), C, labels.size(2), labels.size(3)).to(device).zero_()
    
    target = one_hot.scatter_(1, labels.type(torch.LongTensor).to(device), 1)
    

        
    return target

