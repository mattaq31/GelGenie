from torch import optim as optim


def define_optimizer(optim_weights, lr=1e-4, optimizer_params=None, optimizer_type='Adam'):
    if optimizer_type.lower() == 'adam':
        if optimizer_params is not None:
            beta_1 = optimizer_params['beta_1']
            beta_2 = optimizer_params['beta_2']
            betas = (beta_1, beta_2)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, optim_weights), lr=lr, betas=betas)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, optim_weights), lr=lr)

    elif optimizer_type.lower() == 'rmsprop':
        if optimizer_params is not None:
            alpha = optimizer_params['alpha']
            weight_decay = optimizer_params['weight_decay']
            momentum = optimizer_params['momentum']
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, optim_weights), lr=lr, alpha=alpha,
                                      weight_decay=weight_decay, momentum=momentum)
        else:
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, optim_weights), lr=lr)

    return optimizer


def define_scheduler(base_optimizer, scheduler_type='ReduceLROnPlateau'):
    if scheduler_type == 'ReduceLROnPlateau':
        # goal: maximize Dice score
        learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(base_optimizer, 'max', patience=2)

    # Add this scheduler here:
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler_params = {'t_mult': 1,
                            'restart_period': 10,
                            'lr_min': 1e-7}
        learning_rate_scheduler = \
            optim.lr_scheduler.CosineAnnealingWarmRestarts(base_optimizer, T_mult=scheduler_params['t_mult'],
                                                           T_0=scheduler_params['restart_period'],
                                                           eta_min=scheduler_params['lr_min'])
    else:
        print(f'No scheduler chosen, scheduler = {scheduler_type}')
    return learning_rate_scheduler
