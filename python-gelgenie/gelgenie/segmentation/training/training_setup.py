from torch import optim as optim


def core_setup(network, lr=1e-5, optimizer_type='adam', scheduler_type=None, scheduler_specs=None, **kwargs):

    if optimizer_type == 'rmsprop':
        optimizer = define_optimizer(network.parameters(), lr=lr, optimizer_type='rmsprop',
                                          optimizer_params={'weight_decay': 1e-8, 'momentum': 0.9, 'alpha': 0.99})
    elif optimizer_type == 'adam':
        optimizer = define_optimizer(network.parameters(), lr=lr, optimizer_type='adam')
    else:
        raise RuntimeError('Optimizer not recognized')

    if scheduler_type == 'ReduceLROnPlateau' or scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = define_scheduler(optimizer, scheduler_type=scheduler_type, scheduler_specs=scheduler_specs)
    else:
        scheduler = None

    return optimizer, scheduler


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
    else:
        raise RuntimeError('Optimizer not recognized')

    return optimizer


def define_scheduler(base_optimizer, scheduler_type='ReduceLROnPlateau', scheduler_specs=None):
    if scheduler_type == 'ReduceLROnPlateau':
        learning_rate_scheduler = optim.lr_scheduler.ReduceLROnPlateau(base_optimizer, 'max', patience=2)
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        if scheduler_specs is None:
            scheduler_specs = {'restart_period': 10}
        scheduler_params = {'t_mult': 1,
                            'restart_period': scheduler_specs['restart_period'],
                            'lr_min': 1e-7}
        learning_rate_scheduler = \
            optim.lr_scheduler.CosineAnnealingWarmRestarts(base_optimizer, T_mult=scheduler_params['t_mult'],
                                                           T_0=scheduler_params['restart_period'],
                                                           eta_min=scheduler_params['lr_min'])
    else:
        raise RuntimeError('Scheduler not recognized')

    return learning_rate_scheduler
