import torch
from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model, lr=None):
    lr = cfg.SOLVER.BASE_LR if lr is None else lr
    # return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


def make_lr_scheduler(cfg, optimizer, milestones=None):
    return WarmupMultiStepLR(optimizer=optimizer,
                             milestones=cfg.SOLVER.LR_STEPS if milestones is None else milestones,
                             gamma=cfg.SOLVER.GAMMA,
                             warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                             warmup_iters=cfg.SOLVER.WARMUP_ITERS)


# torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)