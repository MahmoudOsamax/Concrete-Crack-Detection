import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import logging

logger = logging.getLogger(__name__)

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer."""
    try:
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError(f'Normalization layer [{norm_type}] is not found')
        return norm_layer
    except Exception as e:
        logger.error(f"Error in get_norm_layer: {str(e)}")
        raise

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler."""
    try:
        if opt.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        else:
            raise NotImplementedError(f'Learning rate policy [{opt.lr_policy}] is not implemented')
        return scheduler
    except Exception as e:
        logger.error(f"Error in get_scheduler: {str(e)}")
        raise

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    try:
        logger.info(f'Initializing network with {init_type}')
        net.apply(init_func)
    except Exception as e:
        logger.error(f"Error in init_weights: {str(e)}")
        raise

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network."""
    try:
        if gpu_ids:
            assert torch.cuda.is_available(), "CUDA is not available"
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)
        init_weights(net, init_type, init_gain=init_gain)
        logger.info("Network initialized")
        return net
    except Exception as e:
        logger.error(f"Error in init_net: {str(e)}")
        raise