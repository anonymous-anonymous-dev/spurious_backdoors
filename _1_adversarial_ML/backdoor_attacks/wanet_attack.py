import numpy as np
import torch
import torchvision
import os


from _0_general_ML.data_utils.torch_dataset import Torch_Dataset
from _0_general_ML.data_utils.torch_subdataset import Client_SubDataset
from _0_general_ML.model_utils.torch_model import Torch_Model

from utils_.torch_utils import get_data_samples_from_loader, prepare_dataloader_from_numpy
from utils_.general_utils import confirm_directory

from .simple_backdoor import Simple_Backdoor

# from .wanet_official.wanet import train_toy, eval_toy, Parameters
from .wanet_official.wanet_cleaner import Parameters, train_toy, eval_toy, train, eval



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Wanet_Backdoor(Simple_Backdoor):
    
    def __init__(
        self, data: Torch_Dataset, 
        backdoor_configuration=None,
        **kwargs
    ):
        
        super().__init__(data, backdoor_configuration=backdoor_configuration, attack_name='wba')
        
        return
    
    
    def configure_backdoor(self, backdoor_configuration, **kwargs):
        
        super().configure_backdoor(backdoor_configuration, **kwargs)
        
        self.requires_training_control = True
        
        default_configuration= {
            'batch_size': 128,
            'cross_ratio': 2,
            'attack_mode': 'all2one',
            'k': 4,
            's': 0.5,
            'grid_rescale': 1
        }
        for key in default_configuration.keys():
            if key not in self.backdoor_configuration.keys():
                self.backdoor_configuration[key] = default_configuration[key]
        
        (self.input_height, self.input_width) = self.preferred_size
        self.input_channel = self.item.shape[0]
        
        # self.num_classes = self.parent_data.get_output_shape()[0]
        self.num_classes = self.parent_data.num_classes
        
        self.data_means = self.parent_data.data_means
        self.data_stds = self.parent_data.data_stds
        
        self.mode = 'train'
        self.batch_size = self.backdoor_configuration['batch_size']
        # self.pc = 0.1
        # self.device = 'cuda'
        self.cross_ratio = self.backdoor_configuration['cross_ratio']    # rho_a = pc, rho_n = pc * cross_ratio
        self.attack_mode = self.backdoor_configuration['attack_mode']
        # self.schedulerC_lambda = 0.1
        # self.schedulerC_milestones = [100, 200, 300, 400]
        self.k = self.backdoor_configuration['k']
        self.s = self.backdoor_configuration['s']
        # self.n_iters= 1000
        self.grid_rescale = 1
        
        
        # save noise gird and identity to things to save if they are not already there
        grid_name = f'grids_{self.data_name}'
        trigger_folder = '__things_to_save__'
        confirm_directory(trigger_folder)
        folder_filename = f'{trigger_folder}/{grid_name}.npz'
        if os.path.isfile(folder_filename):
            print(f'Loading noise and identity grids.')
            grid_files = np.load(folder_filename)
            self.noise_grid = torch.tensor(grid_files['noise_grid'])
            self.identity_grid = torch.tensor(grid_files['identity_grid'])
        else:
            ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            self.noise_grid = (
                torch.nn.functional.upsample(ins, size=self.input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
            )
            
            array1d = torch.linspace(-1, 1, steps=self.input_height)
            x, y = torch.meshgrid(array1d, array1d)
            self.identity_grid = torch.stack((y, x), 2)[None, ...]#.to(self.device)
            
            print(f'Saving noise and identity grids.')
            np.savez_compressed(folder_filename, noise_grid=self.noise_grid.numpy(), identity_grid=self.identity_grid.numpy())
        
        # grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
        # grid_temps = torch.clamp(grid_temps, -1, 1)
        
        self.compute_poisoned_perturbations_train()
        # self.compute_poisoned_perturbations_test()
        
        return
    
    
    def poison_data(self):
        
        self.poison_ratio = self.backdoor_configuration['poison_ratio']
        if self.backdoor_configuration['poison_ratio_wrt_class_members']:
            self.num_poison_samples = (self.poison_ratio * np.sum([np.sum(np.array(self.train.targets)==target) for target in self.targets])).astype('int')
        else:
            self.num_poison_samples = int(self.poison_ratio * self.train.__len__())
        
        if self.poison_ratio > 0:
            target_indices = np.where(self.train.targets!=self.targets[0])[0]
            self.num_poison_samples = min(self.num_poison_samples, len(target_indices))
            self.poison_indices = np.random.choice(target_indices, size=self.num_poison_samples, replace=False)
            
            self.train.poison_indices = self.poison_indices
            self.train.poisoner_fn = self.poison_train
            self.train.update_targets(self.train.poison_indices, [self.targets[0]]*len(self.train.poison_indices))
            
        self.poisoned_test.poison_indices = np.arange(self.poisoned_test.__len__())
        self.poisoned_test.poisoner_fn = self.poison_test
        self.poisoned_test.update_targets(self.poisoned_test.poison_indices, [self.targets[0]]*len(self.poisoned_test.poison_indices))
        
        return
    
    
    def denormalize(self, inputs, means, stds):
        return (inputs * torch.tensor(stds).view(1, -1, 1, 1)) + torch.tensor(means).view(1, -1, 1, 1)
    def renormalize(self, inputs, means, stds):
        return (inputs - torch.tensor(means).view(1, -1, 1, 1)) / torch.tensor(stds).view(1, -1, 1, 1)
    
    
    def compute_poisoned_perturbations_train(self):
        
        opt = Parameters(self.parent_data)
        
        norm_transform = []
        norm_transform += [torchvision.transforms.Grayscale(3)] if self.data_name=='mnist_3' else []
        norm_transform += [torchvision.transforms.ToTensor()]
        norm_transform += [torchvision.transforms.Normalize(tuple(self.data_means), tuple(self.data_stds))]
        normalization_transform = torchvision.transforms.Compose(norm_transform)
        # print(f'The dataset name is {self.data_name}, channels is {self.input_channel} and transforms are {norm_transform}')
        self.train.update_transforms(normalization_transform)
        self.poisoned_test.update_transforms(normalization_transform)
        
        # self.train.poison_indices = [] # no need for this as this is being called before the poisoning is done.
        train_dl, test_dl = self.prepare_data_loaders(batch_size=32, shuffle=False)

        # # Prepare grid
        # ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
        # ins = ins / torch.mean(torch.abs(ins))
        # noise_grid = (
        #     torch.nn.functional.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
        #     .permute(0, 2, 3, 1)
        #     .to(opt.device)
        # )
        # array1d = torch.linspace(-1, 1, steps=opt.input_height)
        # x, y = torch.meshgrid(array1d, array1d)
        # identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)
        
        # noise_grid = self.noise_grid.to(opt.device)
        # identity_grid = self.identity_grid.to(opt.device)

        train, train_poisoned = train_toy(train_dl, self.noise_grid.to(opt.device), self.identity_grid.to(opt.device), opt)
        test, test_poisoned = eval_toy(test_dl, self.noise_grid.to(opt.device), self.identity_grid.to(opt.device), opt)
        
        m = self.data_means
        s = self.data_stds
        self.perturbations_train = self.denormalize(train_poisoned, m, s)-self.denormalize(train, m, s)
        self.perturbations_test = self.denormalize(test_poisoned, m, s)-self.denormalize(test, m, s)
        
        self.train.reset_transforms()
        self.poisoned_test.reset_transforms()
        # self.train.poison_indices = self.poison_indices
        
        return
    
    
    def __compute_poisoned_perturbations_test(self):
        
        test_dl = torch.utils.data.DataLoader(self.poisoned_test, batch_size=self.batch_size, shuffle=False)
        
        all_inputs, original_inputs, all_targets = [], [], []
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            
            # inputs = self.denormalize(inputs, self.parent_data.data_means, self.parent_data.data_stds)
            
            # Create backdoor data
            # num_bd = int(bs * rate_bd)
            grid_temps = (self.identity_grid + self.s * self.noise_grid / self.input_height) * self.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
            inputs_bd = torch.nn.functional.grid_sample(inputs, grid_temps.repeat(len(inputs), 1, 1, 1), align_corners=True)
            
            targets_bd = torch.ones_like(targets) * self.targets[0]
            
            original_inputs.append(inputs)
            all_inputs.append(inputs_bd); all_targets.append(targets_bd)
            print(f'\rPoisoning test {batch_idx}/{len(test_dl)}', end='')
        print()
        
        original_inputs = torch.cat(original_inputs, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        # all_inputs = self.renormalize(all_inputs, self.parent_data.data_means, self.parent_data.data_stds)
        all_targets = torch.cat(all_targets, dim=0)
        
        self.perturbations_test = all_inputs - original_inputs
        # self.perturbations_test -= torch.mean(self.perturbations_test)
        
        return
    
    
    def poison_train(self, x, y, index=0, **kwargs):
        # return self.train_poisoned[index], self.targets[0]
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x+self.perturbations_train[index], min_value, max_value), self.targets[0]
        
    
    def poison_test(self, x, y, index=0, **kwargs):
        # return self.test_poisoned[index], self.targets[0]
        min_value = torch.min(x) if torch.max(x)>torch.min(x) else 0
        max_value = torch.max(x) if torch.max(x)>torch.min(x) else 1
        return torch.clamp(x+self.perturbations_test[index], min_value, max_value), self.targets[0]
    
    
    def poisoned_eval_shot(
        self, model: Torch_Model,
        verbose: bool=True,
        pre_str: str='', color: str=None,
        **kwargs
    ):
        
        opt = Parameters(self, device=model.device)
        identity_grid = self.identity_grid.to(model.device)
        noise_grid = self.noise_grid.to(model.device)
        
        test_dl = torch.utils.data.DataLoader(self.test, batch_size=model.model_configuration['batch_size'])
        a, b, c, self.poisoned_test_samples_during_eval = eval(model.model, test_dl, noise_grid, identity_grid, 0,0,0, opt, verbose=verbose, pre_str=pre_str, color=color)
        
        return a, b, c
    
    
    def train_shot(
        self, model: Torch_Model,
        epoch: int,
        verbose: bool=True,
        pre_str: str='', color: str=None,
        **kwargs
    ):
        
        opt = Parameters(self, device=model.device)
        identity_grid = self.identity_grid.to(model.device)
        noise_grid = self.noise_grid.to(model.device)
        
        optimizerC = model.optimizer.optim
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
        netC = model.model
        
        train_dl = torch.utils.data.DataLoader(self.parent_data.train, batch_size=model.model_configuration['batch_size'], shuffle=True)
        a, b, c = train(model.model, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, epoch, opt, verbose=verbose, pre_str=pre_str, color=color)
        
        return a, b, c
    
    
    def __train_shot(
        self, model: Torch_Model,
        epoch: int,
        verbose: bool=True,
        pre_str: str='', color: str=None,
        **kwargs
    ):
        
        opt = Parameters(
            self,
            pc=self.poison_ratio/self.num_classes if self.backdoor_configuration['poison_ratio_wrt_class_members'] else self.poison_ratio,
            device=model.device
        )
        
        optimizerC = model.optimizer.optim
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
        netC = model.model
        
        identity_grid = self.identity_grid.to(model.device)
        noise_grid = self.noise_grid.to(model.device)
        
        train_dl = torch.utils.data.DataLoader(self.parent_data.train, batch_size=model.model_configuration['batch_size'])
        
        netC.train()
        rate_bd = opt.pc
        total_loss_ce = 0
        total_sample = 0
        
        total_clean = 0
        total_bd = 0
        total_cross = 0
        total_clean_correct = 0
        total_bd_correct = 0
        total_cross_correct = 0
        criterion_CE = torch.nn.CrossEntropyLoss()
        criterion_BCE = torch.nn.BCELoss()

        # denormalizer = Denormalizer(opt)
        # transforms = PostTensorTransform(opt).to(opt.device)
        total_time = 0

        avg_acc_cross = 0
        
        loss_over_data = 0
        acc_over_data = 0
        all_bd_inputs = []
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            optimizerC.zero_grad()

            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]

            # Create backdoor data
            num_bd = int(bs * rate_bd)
            num_cross = int(num_bd * opt.cross_ratio)
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(num_cross, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = torch.nn.functional.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)

            inputs_cross = torch.nn.functional.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)

            total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
            # total_inputs = transforms(total_inputs)
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
            # start = time()
            total_preds = netC(total_inputs)
            # total_time += time() - start

            loss_ce = criterion_CE(total_preds, total_targets)

            loss = loss_ce
            loss.backward()

            optimizerC.step()
            
            loss_over_data += loss.item()
            pred = total_preds.argmax(1, keepdim=True)
            acc_over_data += pred.eq(total_targets.view_as(pred)).sum().item()
            
            # total_sample += bs
            # total_loss_ce += loss_ce.detach()

            # total_clean += bs - num_bd - num_cross
            # total_bd += num_bd
            # total_cross += num_cross
            # total_clean_correct += torch.sum(
            #     torch.argmax(total_preds[(num_bd + num_cross) :], dim=1) == total_targets[(num_bd + num_cross) :]
            # )
            # total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
            # if num_cross:
            #     total_cross_correct += torch.sum(
            #         torch.argmax(total_preds[num_bd : (num_bd + num_cross)], dim=1)
            #         == total_targets[num_bd : (num_bd + num_cross)]
            #     )
            #     # avg_acc_cross = total_cross_correct * 100.0 / total_cross

            # # avg_acc_clean = total_clean_correct * 100.0 / total_clean
            # # avg_acc_bd = total_bd_correct * 100.0 / total_bd

            # # avg_loss_ce = total_loss_ce / total_sample
            all_bd_inputs.append(inputs_bd.detach().cpu())
            
            if verbose:
                print_str = 'Epoch: {}[{:3.1f}%] | tr_loss: {:.5f} | tr_acc: {:.2f}% | '.format(
                    epoch, 100. * batch_idx / len(train_dl), 
                    loss_over_data / min( (batch_idx+1) * train_dl.batch_size, len(train_dl.dataset) ), 
                    100. * acc_over_data / min( (batch_idx+1) * train_dl.batch_size, len(train_dl.dataset) )
                )
                print('\r' + pre_str + self.update_color_of_str(print_str, color=color), end='')

        schedulerC.step()
        
        netC.eval()
        n_samples = min( len(train_dl)*train_dl.batch_size, len(train_dl.dataset) )
        return loss_over_data/n_samples, acc_over_data/n_samples, self.update_color_of_str(print_str, color=color)
    
    
    