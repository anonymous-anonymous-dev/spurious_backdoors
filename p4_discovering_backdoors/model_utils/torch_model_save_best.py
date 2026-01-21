import copy
import torch
import numpy as np


from _0_general_ML.model_utils.torch_model import Torch_Model

from _1_adversarial_ML.backdoor_attacks.simple_backdoor import Simple_Backdoor



class Torch_Model_Save_Best(Torch_Model):
    
    def __init__(self, data: Simple_Backdoor, model_configuration, path: str = '', **kwargs):
        
        super().__init__(data, model_configuration, path=path, **kwargs)
        
        self.best_test_loss = None; self.best_poisoned_test_loss = None
        self.best_test_acc = None; self.best_poisoned_test_acc = None
        self.best_criteria = None
        
        return
    
    
    def key_flatten_client_state_np(self, client_state_dict: dict):
        
        flattened_client_state = []
        for key in client_state_dict.keys():
            flattened_client_state += client_state_dict[key].clone().cpu().flatten().tolist()
        
        return np.array(flattened_client_state)
    
    
    def key_unflatten_client_state_np(self, flattened_client_state):
        
        client_state_dict_ = copy.deepcopy(self.model.state_dict())
        
        flattened_client_state_copy = torch.tensor(flattened_client_state.copy())
        unflattened_client_state = {}
        for key in client_state_dict_.keys():
            np_state_key = client_state_dict_[key].cpu().numpy()
            unflattened_client_state[key] = flattened_client_state_copy[:len(np_state_key.flatten())].reshape(np_state_key.shape)
            flattened_client_state_copy = flattened_client_state_copy[len(np_state_key.flatten()):]
        
        return unflattened_client_state
    
    
    def train(
        self, start_epoch=1, epochs=1,
        batch_size=64, 
        verbose=True, validate=True,
        save_best_model=True,
        shuffle: bool=True,
        training_type: str='DC',
        **kwargs
    ):
        
        if training_type not in ['DC', 'MR']:
            information = f'Training type {training_type} is not supported. Supported types are "DC" and "MR". Therefore, setting training type to "DC".'
            print(information)
            training_type = 'DC'
        
        train_loader, test_loader = self.data.prepare_data_loaders(batch_size=self.model_configuration['batch_size'], shuffle=shuffle)
        if isinstance(self.data, Simple_Backdoor):
            poisoned_testloader = torch.utils.data.DataLoader(self.data.poisoned_test, batch_size=self.model_configuration['batch_size'], shuffle=shuffle)
        
        self.test_shot(test_loader, verbose=verbose)
        for epoch in range(start_epoch, epochs+1):
            
            if (training_type=='MR') and (self.data.requires_training_control): train_loss, train_acc, train_str = self.data.train_shot(self, epoch, verbose=verbose)
            else: train_loss, train_acc, train_str = self.train_shot(train_loader, epoch, verbose=verbose)
            
            if validate:
                test_loss, test_acc, test_str = self.test_shot(test_loader, verbose=verbose, pre_str=train_str, color='green')
                if isinstance(self.data, Simple_Backdoor):
                    if (training_type=='MR') and (self.data.requires_training_control):
                        p_test_loss, p_test_acc, _ = self.data.poisoned_eval_shot(self, verbose=verbose, pre_str=train_str+test_str, color='light_red')
                    else:
                        p_test_loss, p_test_acc, _ = self.test_shot(poisoned_testloader, verbose=verbose, pre_str=train_str+test_str, color='light_red')
                if save_best_model:
                    criteria = test_acc
                    if isinstance(self.data, Simple_Backdoor):
                        criteria *= p_test_acc
                    if (self.best_criteria is None) or (criteria > self.best_criteria):
                        self.saved_flattened_model = self.key_flatten_client_state_np(self.model.state_dict().copy())
                        # self.best_test_loss = test_loss; self.best_test_acc = test_acc
                        self.best_criteria = criteria
                        print(f' Updating the best model.', end='')
            print()
            
        if save_best_model:
            print('Restoring best model instance.')
            self.model.load_state_dict(self.key_unflatten_client_state_np(self.saved_flattened_model))
        
        return
    
    
    def __restore_best_model(self, *args, **kwargs):
        self.model.load_state_dict(self.key_unflatten_client_state_np(self.saved_flattened_model))
        return
    
    
    def predict_batch_wise(self, x_input: np.ndarray):
        
        batch_size = self.model_configuration['batch_size']
        no_of_batches = int(len(x_input) / batch_size) + 1
        
        predictions = []
        for batch_number in range(no_of_batches):
            start_index = batch_number * batch_size
            end_index = min( (batch_number+1)*batch_size, len(x_input) )
            with torch.no_grad():
                predictions.append( self.model(torch.tensor( x_input[start_index:end_index] ).to(self.device)).cpu() )
        
        return torch.cat(predictions, 0)
    
    