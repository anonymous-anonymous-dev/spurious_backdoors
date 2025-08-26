import numpy as np
import torch, torchvision



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_outputs(
    model_in: torch.nn.Module, data_loader_in: torch.utils.data.DataLoader, 
    device: str=torch_device, return_numpy: bool=False,
    verbose: bool=False, pre_str: str=''
):
    
    model_in.eval()
    with torch.no_grad():
        ac_ = []
        for i, (_x, _y) in enumerate(data_loader_in):
            if verbose:
                print(f'\r{pre_str} | Computing outputs [{100.*(i+1)/len(data_loader_in):.2f} %]:', end='')
            with torch.no_grad():
                ac_.append(model_in(_x.to(device)).cpu())
        ac_ = torch.cat(ac_, 0)
        
    if return_numpy:
        return ac_.detach().cpu().numpy()
    return ac_.cpu()


def get_data_samples_from_loader(data_loader_in: torch.utils.data.DataLoader, size: int=None, return_numpy: bool=False, verbose: bool=False):
    
    # TODO: implement size thing.
    
    data_x, data_y = [], []
    for i, (_x, _y) in enumerate(data_loader_in):
        data_x.append(_x)
        data_y.append(_y)
        
        if verbose: print(f'\rLoading {i+1}/{len(data_loader_in)}.', end='')
    if verbose: print()
        
    if return_numpy:
        return torch.cat(data_x[:size], 0).numpy(), torch.cat(data_y[:size], 0).numpy()
    return torch.cat(data_x[:size], 0), torch.cat(data_y[:size], 0)


def evaluate_on_numpy_arrays(
    _model_in: torch.nn.Module, input_: np.ndarray, classes: np.ndarray, 
    batch_size: int=64,
    device: str=torch_device,
    imagenet_normalize: bool=False,
    use_dataloader: bool=True
):
    
    from .general_utils import normalize
    
    _mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    _std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    
    def _imagenet_normalize(x):
        return (x-_mean)/_std
    
    input_ = _imagenet_normalize(normalize(input_)) if imagenet_normalize else input_
    
    if use_dataloader:
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(input_.astype(np.float32)), torch.tensor(classes)), batch_size=batch_size, shuffle=False
        )
        outputs = get_outputs(_model_in, dataloader, device=device).detach().cpu().numpy()
        predicted_classes = np.argmax(outputs, axis=-1)
    
    else:
        batch_size = len(input_) if batch_size is None else batch_size
        number_of_batches = len(input_) // batch_size
        number_of_batches += 1 if number_of_batches*batch_size < len(input_) else 0
        
        with torch.no_grad():
            predicted_classes = []
            for i in range(number_of_batches):
                predicted_classes.append(np.argmax(_model_in(
                    torch.tensor(input_[i*batch_size:(i+1)*batch_size]).to(device)
                ).detach().cpu().numpy(), axis=-1))
            predicted_classes = np.concatenate(predicted_classes, 0)
    
    return np.mean(predicted_classes == classes)


def prepare_dataloader_from_numpy(x: np.ndarray, y: np.ndarray, transforms: torchvision.transforms.Compose=None, batch_size: int=64, shuffle: bool=False):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), 
        batch_size=batch_size, shuffle=shuffle
    )
    return dataloader


def prepare_dataloader_from_tensor(x, y, transforms: torchvision.transforms.Compose=None, batch_size: int=64, shuffle: bool=False):
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), 
        batch_size=batch_size, shuffle=shuffle
    )
    return dataloader


