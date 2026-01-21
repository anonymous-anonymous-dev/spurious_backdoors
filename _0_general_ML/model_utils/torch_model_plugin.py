import torch
import os
import pandas as pd

from termcolor import colored


from utils_.general_utils import confirm_directory

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset

from _0_general_ML.model_utils.optimizer_utils.torch_optimizer import Torch_Optimizer
from _0_general_ML.model_utils.model_architectures.all_architectures import *



loss_functions = {
    'crossentropy': torch.nn.CrossEntropyLoss(),
    'mse': torch.nn.MSELoss(),
    'nll': torch.nn.NLLLoss(),
    'l1': torch.nn.L1Loss(),
    'kl_div': torch.nn.KLDivLoss(),
    'binary_crossentropy': torch.nn.BCELoss(),
}


local_model_architectures = {
    'mnist_cnn': MNIST_CNN,
    
    'resnet18_gtsrb': resnet18_for_gtsrb,
    'resnet50_gtsrb': Resnet50_GTSRB,
    'cnn_gtsrb': CNN_GTSRB,
    
    'cifar10_vgg11': cifar10_vgg11,
    # 'cifar10_resnet18': Resnet18_CIFAR10,
    'cifar10_resnet18': Resnet18_Custom_CIFAR10,
    # 'cifar10_resnet18': CIFAR10_Resnet18_Backdoor_Bench,
    'cifar10_vit16': ViT16_CIFAR10,
    'cifar10_vit16_official': ViT16_CIFAR10_Official,
    'cifar10_resnet50': Resnet50_CIFAR10,
    'cifar10_convnext': CIFAR10_ConvNeXT,
    'cifar100_convnext': CIFAR100_ConvNeXT,
    
    'cifar100_resnet18': Resnet18_CIFAR100,
    'cifar100_resnet50': Resnet50_CIFAR100,
    'cifar100_vit16': ViT16_CIFAR100,
    
    'cifar4_vgg11': cifar4_vgg11,
    
    'kaggle_imagenet_resnet50': Resnet50_Imagenet,
    'kaggle_imagenet_resnet18': Resnet18_Imagenet,
    'kaggle_imagenet_vit_b_16': ViT_B_Imagenet,
}


def deactivate_batchnorm(m):
    
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()
    
    return


class Torch_Model_Plugin:
    
    def __init__(
        self,
        data: Torch_Dataset, model_configuration: dict,
        path: str='',
        **kwargs
    ):
        
        # let's not copy the dict for now. if we encounter problems will add .copy() at the end.
        self.model_architectures = local_model_architectures
        
        self.data = data
        self.path = path
        
        self.reset_model(model_configuration)
        
        return
    
    
    def update_color_of_str(self, print_str: str, color: str=None):
        if color is not None:
            assert isinstance(color, str), f'The color must be of str type but is {color}.'
        return print_str if color is None else colored(print_str, color)
    
    
    def reset_model(self, model_configuration):
        
        self.model_configuration = {
            'model_architecture': 'mnist_cnn',
            'batch_size': 128,
            'epochs': 1,
            'loss_fn': 'crossentropy',
            'optimizer': 'adam',
            'learning_rate': 1e-4,
            'momentum': 0.5,
            'weight_decay': 0,
            'gpu_number': 0,
            'scheduler': None
        }
        if model_configuration:
            for key in model_configuration.keys():
                self.model_configuration[key] = model_configuration[key]
        
        self.model = self.model_architectures[self.model_configuration['model_architecture']]()
        
        self.device = torch.device('cuda:'+str(self.model_configuration['gpu_number']) if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        optimizer_dict = {}
        optimizer_dict['name'] = self.model_configuration['optimizer']
        optimizer_dict['lr'] = self.model_configuration['learning_rate']
        for key in ['momentum', 'weight_decay']:
            optimizer_dict[key] = self.model_configuration[key]
            
        scheduler_dict = {}
        scheduler_dict['name'] = self.model_configuration['scheduler']
        for key in []:
            scheduler_dict[key] = self.model_configuration[f'scheduler_{key}']
            
        self.optimizer = Torch_Optimizer(self.model.parameters(), optimizer_dict=optimizer_dict, scheduler_dict=scheduler_dict)
        
        # self.optimizer_class = Torch_Optimizer(
        #     name=self.model_configuration['optimizer'],
        #     lr=self.model_configuration['learning_rate'], 
        #     momentum=self.model_configuration['momentum'],
        #     weight_decay=self.model_configuration['weight_decay']
        # )
        # self.optimizer = self.optimizer_class.return_optimizer(self.model.parameters())
        self.loss_function = loss_functions[self.model_configuration['loss_fn']]
        
        self.save_directory = self.path + self.data.data_name + '/'
        self.save_directory += 'torch_models/'
        
        return
    
    
    def save(self, name, save_optimizer=True):
        # confirm_directory(self.save_directory)
        confirm_directory( '/'.join(f'{self.save_directory}{name}'.split('/')[:-1]) )
        self.save_weights(self.save_directory+name, save_optimizer=save_optimizer)
        return
    
    
    def unsave(self, name, **kwargs): return self.unsave_weights(self.save_directory+name)
    
    
    def _deprecated_save(self, name, save_optimizer):
        
        if os.path.isfile(self.path+'directory_of_saved_models.xlsx'):
            df = pd.read_excel(self.path+'directory_of_saved_models.xlsx', engine='openpyxl')
        else:
            df = pd.DataFrame()
        
        n_saved_models = len(os.listdir(self.save_directory))
        df[self.save_directory+name] = n_saved_models
        
        self.save_weights(self.save_directory+str(n_saved_models), save_optimizer=save_optimizer)
        
        return
    
    
    def _deprecated_load(self, name, load_optimizer=False):
        
        df = pd.read_excel(
            self.path+'directory_of_saved_models.xlsx', 
            engine='openpyxl'
        )
        
        if self.save_directory+name in df.columns:
            n_saved_models = df[self.save_directory+name].tolist()[0]
            self.load_weights(self.save_directory+str(n_saved_models))
            return True
        
        return False
        
        
    def load_or_train(
        self, name, 
        epochs=1, batch_size=None, 
        patience=1,
        load_latest_saved_model=False
    ):
        
        self.given_model_name = name
        model_found_and_loaded = self.load_weights(self.save_directory + name)
        
        if not model_found_and_loaded:
            print('Model not found at: ', self.save_directory + name)
            
            latest_model_found, start_epoch = False, 0
            for epoch in range(1, epochs+1):
                latest_model_found = self.load_weights(self.save_directory+name+'_latest({})'.format(epoch))
                if latest_model_found:
                    start_epoch = epoch
            
            print("Training model from scratch.")
            self.train(start_epoch=start_epoch+1, epochs=epochs, batch_size=batch_size, patience=patience)
            self.save(name)

        return
    
    
    def adversarial_train_shot(self, *args, **kwargs): return self.not_implemented()
    
    
    def train_shot(
        self, train_loader, epoch: int,
        verbose: bool=True,
        pre_str: str='', color: str=None,
        **kwargs
    ):
        
        self.model.train()
        
        print_str = ''
        loss_over_data = 0
        acc_over_data = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()
            
            loss_over_data += loss.item()
            pred = output.argmax(1, keepdim=True)
            acc_over_data += pred.eq(target.view_as(pred)).sum().item()
            
            if verbose:
                print_str = 'Epoch: {}[{:3.1f}%] | tr_loss: {:.5f} | tr_acc: {:.2f}% | '.format(
                    epoch, 100. * batch_idx / len(train_loader), 
                    loss_over_data / min( (batch_idx+1) * train_loader.batch_size, len(train_loader.dataset) ), 
                    100. * acc_over_data / min( (batch_idx+1) * train_loader.batch_size, len(train_loader.dataset) )
                )
                print('\r' + pre_str + self.update_color_of_str(print_str, color=color), end='')
        
        self.model.eval()
        
        n_samples = min( len(train_loader)*train_loader.batch_size, len(train_loader.dataset) )
        return loss_over_data/n_samples, acc_over_data/n_samples, self.update_color_of_str(print_str, color=color)
    
    
    def test_shot(self, test_loader, verbose: bool=True, pre_str: str='', color: str=None, **kwargs):
        
        self.model.eval()
        
        print_str = ''
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                test_loss += self.loss_function(output, target).item()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
                if verbose:
                    print_str = '({:3.1f}%) ts_loss: {:.9f} | ts_acc: {:.2f}% | '
                    print_str = print_str.format(
                        100. * (batch_idx+1) / len(test_loader), 
                        test_loss / min( (batch_idx+1) * test_loader.batch_size, len(test_loader.dataset) ), 
                        100. * correct / min( (batch_idx+1) * test_loader.batch_size, len(test_loader.dataset) )
                    )
                    print('\r' + pre_str + self.update_color_of_str(print_str, color=color), end='')
        
        n_samples = min( len(test_loader)*test_loader.batch_size, len(test_loader.dataset) )
        return test_loss/n_samples, correct/n_samples, self.update_color_of_str(print_str, color=color)
    
    
    def predict(self, test_loader, verbose=True, pre_str='', post_str='', **kwargs):
        
        self.model.eval()
        
        outputs, ground_truths = [], []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                outputs.append( self.model(data.to(self.device)) )
                ground_truths.append( target.to(self.device) )
        
                if verbose:
                    print_str = '({:3.1f}%)'.format( 100.*(batch_idx+1)/len(test_loader) )
                    print('\r' + pre_str + print_str + post_str, end='')
                            
        return torch.cat(outputs, dim=0), torch.cat(ground_truths, dim=0)
    
    
    def not_implemented(self, *args, **kwargs):
        raise NotImplementedError('This class does not support the called function.')
    
    
    def get_modules(self, super_module: torch.nn.Module):
        modules_dict = dict(super_module._modules)
        for name in modules_dict.keys():
            # print(name)
            # print(module.requires_grad)
            # print(name)
            name_dict = dict(modules_dict[name]._modules)
            if len(name_dict)>0:
                self.get_modules(modules_dict[name])
            else:
                print(isinstance(modules_dict[name], torch.nn.Linear))
        return
    
    
    def get_children(self, super_module: torch.nn.Module=None):
        super_module = self.model if super_module is None else super_module
        modules_list = list(super_module.children())
        final_modules_list = []
        for module in modules_list:
            name_list = list(module.children())
            if len(name_list)>0:
                final_modules_list += self.get_children(module)
            else:
                final_modules_list += [module]
        return final_modules_list
    
    
    def freeze_non_linear_unfreeze_linear(self, super_module: torch.nn.Module):
        modules_list = list(super_module.children())
        for module in modules_list:
            name_list = list(module.children())
            if len(name_list)>0:
                self.freeze_non_linear_unfreeze_linear(module)
            else:
                if not isinstance(module, torch.nn.Linear):
                    # if not isinstance(module, torch.nn.Dropout):
                    for param in module.parameters():
                        param.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = True
        return
    
    
    def unfreeze_last_n_layers(self, super_module: torch.nn.Module=None, n: int=3):
        modules_list = self.get_children(super_module if super_module is not None else self.model)
        interesting_modules = modules_list[-n:] if isinstance(n, int) else modules_list
        for module in interesting_modules:
            # print(f'Changing module: {module}')
            for param in module.parameters():
                param.requires_grad = True
        return


    def freeze_last_n_layers(self, super_module: torch.nn.Module=None, n: int=3):
        modules_list = self.get_children(super_module if super_module is not None else self.model)
        interesting_modules = modules_list[-n:] if isinstance(n, int) else modules_list
        for module in interesting_modules:
            # print(f'Changing module: {module}')
            for param in module.parameters():
                param.requires_grad = False
        return


    # def make_non_linear_layers_non_trainable(super_module):
    def freeze_last_linear(self, super_module: torch.nn.Module=None):
        modules_list = self.get_children(super_module if super_module is not None else self.model)
        for param in modules_list[-1].parameters():
            param.requires_grad = False
        return
    
    