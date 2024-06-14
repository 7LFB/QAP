import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import scipy.stats as stats2
import sys

eps = torch.finfo(torch.float32).eps



class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #first compute cross-entropy
        inputs = F.softmax(inputs,dim=-1)  
        BCE = F.cross_entropy(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        focal_loss = torch.mean(focal_loss)
                       
        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky



class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky


class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)    
        Lovasz = lovasz_hinge(inputs, targets, per_image=False)                       
        return Lovasz


#PyTorch
# ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
# beta = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, e, 1.0 - e)       
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (beta * weighted_ce) - ((1 - beta) * dice)
        
        return combo


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU




# ------------- for NMF -------------------------
def kl_div(input: Tensor, target: Tensor) -> Tensor:
    r"""The generalized `Kullback-Leibler divergence Loss
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__, which equal to β-divergence loss when β = 1.

    The loss can be described as:

    .. math::
        \ell(x, y) = \sum_{n = 0}^{N - 1} x_n log(\frac{x_n}{y_n}) - x_n + y_n

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input

    Returns:
        Tensor: single element tensor
    """
    return target.reshape(-1) @ (target.add(eps).log() - input.add(eps).log()).reshape(-1) - target.sum() + input.sum()


def euclidean(input: Tensor, target: Tensor) -> Tensor:
    r"""The `Euclidean distance
    <https://en.wikipedia.org/wiki/Euclidean_distance>`__, which equal to β-divergence loss when β = 2.

    .. math::
        \ell(x, y) = \frac{1}{2} \sum_{n = 0}^{N - 1} (x_n - y_n)^2

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input

    Returns:
        Tensor: single element tensor
    """
    return F.mse_loss(input, target, reduction='sum') * 0.5


def is_div(input: Tensor, target: Tensor) -> Tensor:
    r"""The `Itakura–Saito divergence
    <https://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance>`__, which equal to β-divergence loss when β = 0.

    .. math::
        \ell(x, y) = \sum_{n = 0}^{N - 1} \frac{x_n}{y_n} -  log(\frac{x_n}{y_n}) - 1

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input

    Returns:
        Tensor: single element tensor
    """
    target_eps, input_eps = target.add(eps), input.add(eps)
    return (target_eps / input_eps).sum() - target_eps.log().sum() + input_eps.log().sum() - target.numel()


def beta_div(input, target, beta=0):
    r"""The `β-divergence loss
    <https://arxiv.org/pdf/1010.1763.pdf>`__ measure

    The loss can be described as:

    .. math::
        \ell(x, y) = \sum_{n = 0}^{N - 1} \frac{1}{\beta (\beta - 1)}\left ( x_n^{\beta} + \left (\beta - 1
        \right ) y_n^{\beta} - \beta x_n y_n^{\beta-1}\right )

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input
        beta (float): a real value control the shape of loss function

    Returns:
        Tensor: single element tensor
    """
    if beta == 2:
        return euclidean(input, target)
    elif beta == 1:
        return kl_div(input, target)
    elif beta == 0:
        return is_div(input, target)
    else:
        input = input.reshape(-1).add(eps)
        target = target.reshape(-1)
        if beta < 0:
            target = target.add(eps)
        bminus = beta - 1

        term_1 = target.pow(beta).sum()
        term_2 = input.pow(beta).sum()
        term_3 = target @ input.pow(bminus)

        loss = term_1 + bminus * term_2 - beta * term_3
        return loss / (beta * bminus)


def sparseness(x: Tensor) -> Tensor:
    r"""The sparseness measure proposed in
    `Non-negative Matrix Factorization with Sparseness Constraints
    <https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf>`__, can be caculated as:

    .. math::
        f(x) = \frac{\sqrt{N} - \frac{\sum_{n=0}^{N-1} |x_n|}{\sqrt{\sum_{n=0}^{N-1} x_n^2}}}{\sqrt{N} - 1}


    Args:
        x (Tensor): tensor of arbitrary shape

    Returns:
        Tensor: single element tensor with value range between 0 (the most sparse) to 1 (the most dense)
    """
    N = x.numel()
    return (N ** 0.5 - x.norm(1) / x.norm(2)) / (N ** 0.5 - 1)



def get_gauss_label(label, n_classes, amplifier, noise=0):
    n = n_classes*amplifier
    half_int = amplifier/2
    label_noise = np.random.uniform(low=-noise, high=noise)
    if label == 0:
        label_noise = np.abs(label_noise)
    if label == 4:
        label_noise = -np.abs(label_noise)
    label += label_noise
    label_new = half_int + label*amplifier
    gauss_label = stats2.norm.pdf(np.arange(n), label_new, half_int/2)
    gauss_label/=np.sum(gauss_label)
    return gauss_label

def get_gaussian_label_distribution(n_classes, std=0.5):
    cls = []
    for n in range(n_classes):
        cls.append(stats2.norm.pdf(range(n_classes), n, std))
    dists = np.stack(cls, axis=0)
    return dists
    # if n_classes == 3:
    #     CL_0 = stats2.norm.pdf([0, 1, 2], 0, std)
    #     CL_1 = stats2.norm.pdf([0, 1, 2], 1, std)
    #     CL_2 = stats2.norm.pdf([0, 1, 2], 2, std)
    #     dists = np.stack([CL_0, CL_1, CL_2], axis=0)
    #     return dists
    # if n_classes == 5:
    #     CL_0 = stats2.norm.pdf([0, 1, 2, 3, 4], 0, std)
    #     CL_1 = stats2.norm.pdf([0, 1, 2, 3, 4], 1, std)
    #     CL_2 = stats2.norm.pdf([0, 1, 2, 3, 4], 2, std)
    #     CL_3 = stats2.norm.pdf([0, 1, 2, 3, 4], 3, std)
    #     CL_4 = stats2.norm.pdf([0, 1, 2, 3, 4], 4, std)
    #     dists = np.stack([CL_0, CL_1, CL_2, CL_3, CL_4], axis=0)
    #     return dists
    # if n_classes == 6:
    #     CL_0 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 0, std)
    #     CL_1 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 1, std)
    #     CL_2 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 2, std)
    #     CL_3 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 3, std)
    #     CL_4 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 4, std)
    #     CL_5 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 5, std)
    #     dists = np.stack([CL_0, CL_1, CL_2, CL_3, CL_4, CL_5], axis=0)
    #     return dists
    # else:
    #     raise NotImplementedError

def cross_entropy_loss_one_hot(logits, target, reduction='mean'):
    logp = F.log_softmax(logits, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

def one_hot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)

def label_smoothing_criterion(alpha=0.1, distribution='uniform', std=0.5, reduction='mean'):
    def _label_smoothing_criterion(logits, labels):
        n_classes = logits.size(1)
        device = logits.device
        # manipulate labels
        one_hot = one_hot_encoding(labels, n_classes).float().to(device)
        if distribution == 'uniform':
            uniform = torch.ones_like(one_hot).to(device)/n_classes
            soft_labels = (1 - alpha)*one_hot + alpha*uniform
        elif distribution == 'gaussian':
            dist = get_gaussian_label_distribution(n_classes, std=std)
            soft_labels = torch.from_numpy(dist[labels.cpu().numpy()]).to(device)
        else:
            raise NotImplementedError
        loss = cross_entropy_loss_one_hot(logits, soft_labels.float(), reduction)

        return loss

    return _label_smoothing_criterion

def cost_sensitive_loss(input, target, M):
    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))
    device = input.device
    M = M.to(device)
    return (M[target, :]*input.float()).sum(axis=-1)
    # return torch.diag(torch.matmul(input, M[:, target]))

class CostSensitiveLoss(nn.Module):
    def __init__(self,  n_classes, exp=1, normalization='softmax', reduction='mean'):
        super(CostSensitiveLoss, self).__init__()
        if normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        elif normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = None
        self.reduction = reduction
        x = np.abs(np.arange(n_classes, dtype=np.float32))
        M = np.abs((x[:, np.newaxis] - x[np.newaxis, :])) ** exp
        M /= M.max()
        self.M = torch.from_numpy(M)

    def forward(self, logits, target):
        preds = self.normalization(logits)
        loss = cost_sensitive_loss(preds, target, self.M)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError('`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

class CostSensitiveRegularizedLoss(nn.Module):
    def __init__(self,  n_classes=5, exp=2, normalization='softmax', reduction='mean', base_loss='ce', lambd=10):
        super(CostSensitiveRegularizedLoss, self).__init__()
        if normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        elif normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = None
        self.reduction = reduction
        x = np.abs(np.arange(n_classes, dtype=np.float32))
        M = np.abs((x[:, np.newaxis] - x[np.newaxis, :])) ** exp
        #
        # M_oph = np.array([
        #                 [1469, 4, 5,  0,  0],
        #                 [58, 62,  5,  0,  0],
        #                 [22, 3, 118,  1,  0],
        #                 [0, 0,   13, 36,  1],
        #                 [0, 0,    0,  1, 15]
        #                 ], dtype=np.float)
        # M_oph = M_oph.T
        # # Normalize M_oph to obtain M_difficulty:
        # M_difficulty = 1-np.divide(M_oph, np.sum(M_oph, axis=1)[:, None])
        # # OPTION 1: average M and M_difficulty:
        # M = 0.5 * M + 0.5 * M_difficulty
        # ################
        # # OPTION 2: replace uninformative entries in M_difficulty by entries of M:
        # # M_difficulty[M_oph == 0] = M[M_oph == 0]
        # # M = M_difficulty

        M /= M.max()
        self.M = torch.from_numpy(M)
        self.lambd = lambd
        self.base_loss_type = base_loss

        if self.base_loss_type == 'ce':
            self.base_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        elif self.base_loss_type == 'ls':
            self.base_loss = label_smoothing_criterion(distribution='uniform', reduction=reduction)
        elif self.base_loss_type == 'gls':
            self.base_loss = label_smoothing_criterion(distribution='gaussian', reduction=reduction)
        elif self.base_loss_type == 'focal':
            kwargs = {"alpha": 0.8, "gamma": 2.0, "reduction": reduction}
            # self.base_loss = focal_loss(**kwargs) % something wrong with gradients
            self.base_loss = FocalLoss()
        else:
            sys.exit('not a supported base_loss')

    def forward(self, logits, target):

        if self.base_loss_type == 'focal':
            logits=logits.unsqueeze(-1).unsqueeze(-1) # N x C x H x W
            target=target.unsqueeze(-1).unsqueeze(-1) # N x H x W

        base_l = self.base_loss(logits, target)
        if self.lambd == 0:
            return self.base_loss(logits, target)
        else:
            preds = self.normalization(logits)
            loss = cost_sensitive_loss(preds, target, self.M)
            if self.reduction == 'none':
                return base_l + self.lambd*loss
            elif self.reduction == 'mean':
                return base_l + self.lambd*loss.mean()
            elif self.reduction == 'sum':
                return base_l + self.lambd*loss.sum()
            else:
                raise ValueError('`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

def get_cost_sensitive_criterion(n_classes=5, exp=2):
    train_criterion = CostSensitiveLoss(n_classes, exp=exp, normalization='softmax')
    val_criterion = CostSensitiveLoss(n_classes, exp=exp, normalization='softmax')
    return train_criterion, val_criterion

def get_cost_sensitive_regularized_criterion(base_loss='ce', n_classes=5, lambd=1, exp=2):
    train_criterion = CostSensitiveRegularizedLoss(n_classes, exp=exp, normalization='softmax', base_loss=base_loss, lambd=lambd)
    val_criterion = CostSensitiveRegularizedLoss(n_classes, exp=exp, normalization='softmax', base_loss=base_loss, lambd=lambd)

    return train_criterion, val_criterion


class DynamicCostSensitiveRegularizedLoss(nn.Module):
    def __init__(self,  n_classes=5, normalization='softmax', reduction='mean', base_loss='ce', M_hard=0):
        super(DynamicCostSensitiveRegularizedLoss, self).__init__()
        if normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        elif normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = None
        self.reduction = reduction

        self.n_classes=n_classes
    
        self.base_loss_type = base_loss
        self.M_hard = M_hard
        x = np.abs(np.arange(self.n_classes, dtype=np.float32))
        if self.M_hard:
            self.M_init = torch.from_numpy(np.array([[0,self.M_hard/4,2/4,3/4,4/4],
                                                    [1/3,0,1/3,2/3,3/3],
                                                    [2/2,1/2,0,1/2,2/2],
                                                    [3/3,2/3,1/3,0,1/3],
                                                    [4/4,3/4,2/4,self.M_hard/4,0],])).cuda()

        else:   
            self.M_init = torch.from_numpy(np.abs((x[:, np.newaxis] - x[np.newaxis, :]))).cuda()


        if self.base_loss_type == 'ce':
            self.base_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        elif self.base_loss_type == 'ls':
            self.base_loss = label_smoothing_criterion(distribution='uniform', reduction=reduction)
        elif self.base_loss_type == 'gls':
            self.base_loss = label_smoothing_criterion(distribution='gaussian', reduction=reduction)
        elif self.base_loss_type == 'focal':
            kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": reduction}
            self.base_loss = FocalLoss()
        else:
            sys.exit('not a supported base_loss')

    def forward(self, logits, target, exp=2, lambd=10):

        M =self.M_init

        M = M**exp
    
        M /= M.max()


        # if self.base_loss_type == 'focal': % affect gradients
        #     logits=logits.unsqueeze(-1).unsqueeze(-1) # N x C x H x W
        #     target=target.unsqueeze(-1).unsqueeze(-1) # N x H x W

        base_l = self.base_loss(logits, target)
        if lambd == 0:
            return self.base_loss(logits, target)
        else:
            preds = self.normalization(logits)
            loss = cost_sensitive_loss(preds, target, M)
            if self.reduction == 'none':
                return base_l + lambd*loss
            elif self.reduction == 'mean':
                # print(f'baseLoss: {base_l:3f} regLoss{loss.mean():.3f}\t')
                return base_l + lambd*loss.mean()
            elif self.reduction == 'sum':
                return base_l + lambd*loss.sum()
            else:
                raise ValueError('`reduction` must be one of \'none\', \'mean\', or \'sum\'.')