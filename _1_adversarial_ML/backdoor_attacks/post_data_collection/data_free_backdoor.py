import numpy as np
import torch
from copy import deepcopy

from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset, Client_Torch_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model
from _0_general_ML.data_utils.datasets import Fashion_MNIST, Fashion_MNIST_3, GTSRB

from ..simple_backdoor import Simple_Backdoor

from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy, get_outputs



class DataFree_Backdoor(Simple_Backdoor):
    
    def __init__(self, data: Torch_Dataset, backdoor_configuration: dict={}, model: Torch_Model=None, attack_name: str='dfba', **kwargs):
        
        assert model is not None, f'A trained model must be present for the post-training, model-level backdoor attacks.'
        
        self.torch_model = model
        self.poisoned_model = None
        
        self.num_classes = self.torch_model.data.num_classes
        self.limited_data = deepcopy(data)
        self.limited_data.test = Client_SubDataset(data.test, indices=np.random.choice(data.test.__len__(), size=3, replace=False))
        self.limited_data.train = self.limited_data.test
        
        # variables to shift to dictionary later
        self.reduced_size = 40000
        self.epochs = 100
        self.tau_0 = 0.999
        self.tau_1 = 0.800
        self.alpha = 5e-2
        self.batch_size = self.torch_model.model_configuration['batch_size']
        
        self.free_data = GTSRB(preferred_size=data.preferred_size, data_means=data.data_means, data_stds=data.data_stds)
        if data.channels==1:
            self.free_data = Fashion_MNIST(preferred_size=data.preferred_size, data_means=data.data_means, data_stds=data.data_stds)
        self.reduce_data()
        
        super().__init__(self.all_data, backdoor_configuration, attack_name, **kwargs)
        self.poison_model()
        self.poisoned_model.model.load_state_dict(self.key_unflatten_client_state_np(self.best_current_model_np))
        
        return
    
    
    def reduce_data(self):
        
        # give the self.torch_model reduce self.free data by selecting a total of 'k' samples
        # 
        # Let's say we have a total of 'n' batches of self.free data
        # measure the cosine similarity of inputs in a batch: cos_1
        # also measure the cosine similarlity of output logits in a batch: cos_2
        # select those 'k/n' samples that show lease cos_1 * cos_2 in the batch
        # 
        # After the samples are selected, divide them into two parts by 0.8/0.2 ratio — one is self.free_train and the other is self.free_test
        # Poison p% samples in self.free_train to get self.free_poisoned_train and 100% in self.free_test to get self.free_poisoned_test.
        
        all_dl = torch.utils.data.DataLoader(self.free_data.train, batch_size=self.batch_size, shuffle=False)
        # self.free_data.train.targets = np.argmax(get_outputs(self.torch_model.model, all_dl, return_numpy=True), axis=1).tolist()
        _targets = np.random.randint(0, self.torch_model.data.num_classes, size=(len(self.free_data.train.targets))).astype('int')
        _targets[_targets==0] = (_targets[_targets==0]+1) % self.torch_model.data.num_classes
        self.free_data.train.targets = _targets.tolist()
        
        # x, y = get_data_samples_from_loader(all_dl, return_numpy=True)
        num_samples_from_each_batch = 1 + (self.reduced_size // (self.free_data.train.__len__() // self.batch_size))
        
        indices = []
        for _i, (_x, _y) in enumerate(all_dl):
            _x = _x.to(self.torch_model.device)
            _out = self.torch_model.model(_x)
            cos_1 = self.pairwise_cosine_similarity_torch(_x.view(len(_x), -1))
            cos_2 = self.pairwise_cosine_similarity_torch(_out.view(len(_out), -1))
            _cos = torch.mean(torch.abs(cos_1 * cos_2), axis=1).detach().cpu().numpy()
            _best_indices = np.argsort(_cos)[:num_samples_from_each_batch].tolist()
            indices += _best_indices
            
        self.all_data = deepcopy(Client_Torch_SubDataset(self.free_data, indices, train_size=0.8))
        self.all_data.num_classes = self.torch_model.data.num_classes
        
        return
    
    
    def configure_backdoor(self, backdoor_configuration, **kwargs):
        
        super().configure_backdoor(backdoor_configuration, **kwargs)
        
        self.free_poisoned_train_dl = torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        self.free_test_dl = torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
        self.free_poisoned_test_dl = torch.utils.data.DataLoader(self.poisoned_test, batch_size=self.batch_size, shuffle=False)
        
        return
    
    
    def pairwise_cosine_similarity_torch(self, first_state: torch.Tensor, second_state: torch.Tensor=None):
        
        normalized_input_a = torch.nn.functional.normalize(first_state)
        normalized_input_b = normalized_input_a if second_state is None else torch.nn.functional.normalize(second_state)
        res = torch.mm(normalized_input_a, normalized_input_b.T)
        res[res==0] = 1e-6
        
        return res
    
    
    def __poison(self, x, y, **kwargs):
        xp, yp = super().poison(x, y, **kwargs)
        return xp, torch.nn.functional.one_hot(self.targets[0], num_classes=self.num_classes)
    
    
    def key_flatten_client_state_np(self, client_state_dict: dict):
        
        flattened_client_state = []
        for key in client_state_dict.keys():
            flattened_client_state += client_state_dict[key].clone().cpu().flatten().tolist()
        
        return np.array(flattened_client_state)
    
    
    def key_unflatten_client_state_np(self, flattened_client_state):
        
        client_state_dict_ = deepcopy(self.torch_model.model.state_dict())
        
        flattened_client_state_copy = torch.tensor(flattened_client_state.copy())
        unflattened_client_state = {}
        for key in client_state_dict_.keys():
            np_state_key = client_state_dict_[key].cpu().numpy()
            unflattened_client_state[key] = flattened_client_state_copy[:len(np_state_key.flatten())].reshape(np_state_key.shape)
            flattened_client_state_copy = flattened_client_state_copy[len(np_state_key.flatten()):]
        
        return unflattened_client_state
    
    
    def poison_model(self):
        
        print('Poisoning the model.')
        self.poison_target = torch.nn.functional.one_hot(torch.tensor([self.targets[0]]), num_classes=self.num_classes)[0]
        
        # self.poisoned_model = deepcopy(self.torch_model)
        self.poisoned_model = Torch_Model(self.torch_model.data, self.torch_model.model_configuration, path=self.torch_model.path)
        self.poisoned_model.model.load_state_dict(self.torch_model.model.state_dict())
        
        self.poisoned_model.freeze_last_n_layers(n=None)
        self.poisoned_model.unfreeze_last_n_layers(n=7)
        self.poisoned_model.freeze_last_n_layers(n=4)
        # self.poisoned_model.data = self
        
        self.lambda_1 = 1
        self.best_p0_p1 = 0.
        for i in range(self.epochs):
            
            clean_data_clean_model = get_outputs(self.torch_model.model, self.free_test_dl, return_numpy=True); clean_data_clean_model = np.nan_to_num(clean_data_clean_model, nan=0.0)
            clean_data_poisoned_model = get_outputs(self.poisoned_model.model, self.free_test_dl, return_numpy=True); clean_data_poisoned_model = np.nan_to_num(clean_data_poisoned_model, nan=0.0)
            poisoned_data_poisoned_model = get_outputs(self.poisoned_model.model, self.free_poisoned_test_dl, return_numpy=True); poisoned_data_poisoned_model = np.nan_to_num(poisoned_data_poisoned_model, nan=0.)
            
            # calculate cosine similarity of self.torch_model(self.free_test) and self.poisoned_model(self.free_test)
            p_0 = cosine_similarity(clean_data_clean_model.reshape(len(clean_data_clean_model), -1), clean_data_poisoned_model.reshape(len(clean_data_poisoned_model), -1))[0][0]
            # calculate the ASR of self.poisoned_model on self.free_poisoned_test
            p_1 = np.mean(np.argmax(poisoned_data_poisoned_model, axis=1)==self.targets[0])
            if p_0*p_1>self.best_p0_p1:
                self.best_current_model_np = self.key_flatten_client_state_np(self.poisoned_model.model.state_dict())
                self.best_p0_p1 = p_0 * p_1
            
            print(f'Epoch {i}: P_0={p_0:.4f}, P_1={p_1:.4f}, Lambda_1={self.lambda_1:.4f}')
            if (p_0>self.tau_0) and (p_1>self.tau_1):
                return
            
            self.lambda_1 = np.clip(self.lambda_1 + self.alpha * (p_0-p_1), 0, 1)
            # self.poisoned_model.train_shot(self.free_poisoned_train_dl, i, verbose=True)
            self.train_shot_model(self.poisoned_model)
        
        return
    
    
    def train_shot_model(self, model: Torch_Model, data: Torch_Dataset=None):
        
        train_loader = self.free_poisoned_train_dl if data is None else torch.utils.data.DataLoader(data, batch_size=self.batch_size)
        
        model.model.train()
        loss_function_1 = torch.nn.MSELoss()
        loss_function = torch.nn.CosineSimilarity(dim=1)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(model.device), target.to(model.device)
            targets_ = 5 * (torch.stack([self.poison_target]*len(target), dim=0)-0.5).to(model.device)
            # targets_ = torch.tensor([self.targets[0]]*len(target)).to(model.device)
            
            model.optimizer.zero_grad()
            output = model.model(data)
            output_1 = self.torch_model.model(data)
            
            non_target_indices = (target!=self.targets[0])
            target_indices = (target==self.targets[0])
            
            loss_clean = 0
            # loss_clean += loss_function(output[non_target_indices], output_1[non_target_indices]) +
            loss_clean += loss_function_1(output[non_target_indices], output_1[non_target_indices])
            loss_poisoned = loss_function_1(output[target_indices], targets_[target_indices])
            loss = 0
            loss += loss_clean if np.sum(non_target_indices.detach().cpu().numpy())>0 else 0
            loss += self.lambda_1 * loss_poisoned if np.sum(target_indices.detach().cpu().numpy())>0 else 0
            loss = loss.mean()
            
            loss.backward()
            model.optimizer.step()
        
        model.model.eval()
        
        return model
    
    
    def __poison_data(self):
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
        
        if self.poison_ratio > 0:
            # target_indices = np.arange(self.train.__len__())
            target_indices = np.where(self.train.targets!=self.targets[0])[0]
            self.num_poison_samples = min(self.num_poison_samples, len(target_indices))
            self.poison_indices = np.random.choice(target_indices, size=self.num_poison_samples, replace=False)
            
            self.train.poison_indices = self.poison_indices
            self.train.poisoner_fn = self.poison
            self.train.update_targets(self.train.poison_indices, [self.targets[0]]*len(self.train.poison_indices))
            
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison
        self.poisoned_test.update_targets(self.poisoned_test.poison_indices, [self.targets[0]]*len(self.poisoned_test.poison_indices))
        
        return
    
    
    