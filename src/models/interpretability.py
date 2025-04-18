import torch

def compute_feature_importance(model, inputs, target, loss_fn):
    """
    Computes feature importance via input gradients.
    """
    inputs = {k: v.clone().detach().requires_grad_(True) for k, v in inputs.items()}
    output = model(inputs)
    loss = loss_fn(output, target)
    loss.backward()
    importances = {k: v.grad.abs().mean(dim=0).cpu().numpy() for k, v in inputs.items()}
    return importances

def visualize_attention(attn_weights, ax=None):
    """
    Visualizes attention weights as a heatmap.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    cax = ax.matshow(attn_weights, cmap='viridis')
    plt.colorbar(cax, ax=ax)
    ax.set_title("Attention Weights")
    plt.show()
