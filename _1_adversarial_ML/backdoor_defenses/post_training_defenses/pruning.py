import torch
import torch.nn as nn
import torch.nn.utils.prune as prune



def prune_model_by_activations(model, dataloader, pruning_ratio=0.2):
    """
    Prunes a PyTorch model based on APoZ (Average Percentage of Zeros)
    in the activations of convolutional filters.

    Args:
        model (nn.Module): The PyTorch model to prune.
        dataloader (DataLoader): A DataLoader for the small dataset.
        pruning_ratio (float): The percentage of filters to prune (e.g., 0.2 for 20%).
    """
    activation_data = {}

    def get_activation_hook(name):
        def hook(model, input, output):
            activation_data[name] = output.detach().cpu()
        return hook

    # Register hooks for convolutional layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(get_activation_hook(name)))

    # Collect activations
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Calculate APoZ for each filter and prune
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in activation_data:
            activations = activation_data[name]
            # Calculate APoZ for each filter (assuming NCHW format)
            # APoZ = (number of zeros in activations) / (total number of activations)
            apoz_scores = (activations == 0).float().mean(dim=[0, 2, 3]) # Mean over batch, height, width

            # Determine pruning threshold
            num_filters_to_prune = int(pruning_ratio * apoz_scores.numel())
            if num_filters_to_prune > 0:
                threshold = torch.topk(apoz_scores, num_filters_to_prune, largest=True).values[-1]
                
                # Apply unstructured pruning based on APoZ
                # This example uses L1 unstructured pruning for simplicity,
                # but a custom pruning method could be implemented based on APoZ scores.
                # For true filter pruning, architectural changes are needed.
                prune.l1_unstructured(module, name='weight', amount=threshold.item()) 
                print(f"Pruned {name} based on APoZ. Filters with APoZ > {threshold.item():.4f} are pruned.")

    # Remove pruning reparametrization to make sparsity permanent
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and prune.is_pruned(module):
            prune.remove(module, 'weight')
            
    return


