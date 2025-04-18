import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def get_scheduler(optimizer, scheduler_type='plateau', **kwargs):
    if scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
