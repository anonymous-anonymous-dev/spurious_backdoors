import json
import os
import shutil
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from termcolor import colored

from _0_general_ML.model_utils.torch_model import Torch_Model
from utils_.general_utils import normalize
from utils_.visual_utils import show_image_grid

# from classifier_models import PreActResNet18, ResNet18
# import config
from .models import Denormalizer, Normalizer
from .parameters import Parameters
from torch import nn



def update_color_of_str(print_str: str, color: str=None):
    if color is not None:
        assert isinstance(color, str), f'The color must be of str type but is {color}.'
    return print_str if color is None else colored(print_str, color)


def train_toy(
    train_dl, 
    noise_grid, 
    identity_grid, 
    opt: Parameters
):
    
    rate_bd = 1
    
    all_bd_inputs, actual_inputs = [], []
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        
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

        inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)

        inputs_cross = F.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)
        
        all_bd_inputs.append(inputs_bd.detach().cpu())
        actual_inputs.append(inputs.detach().cpu())
        
    return torch.cat(actual_inputs, dim=0), torch.cat(all_bd_inputs, dim=0)


def eval_toy(
    test_dl,
    noise_grid,
    identity_grid,
    opt: Parameters,
):
    
    total_sample = 0
    
    actual_inputs, all_bd_test_inputs = [], []
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            
            actual_inputs.append(inputs.detach().cpu())
            all_bd_test_inputs.append(inputs_bd.detach().cpu())

    return torch.cat(actual_inputs, dim=0), torch.cat(all_bd_test_inputs, dim=0)


def train(
    netC, 
    optimizerC, 
    schedulerC, 
    train_dl, 
    noise_grid, 
    identity_grid, 
    # tf_writer, 
    epoch, 
    opt: Parameters,
    verbose: bool=True, pre_str: str='', color: str=None
):
    
    # print(" Train:")
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

    denormalizer = Denormalizer(opt)
    # transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    avg_acc_cross = 0
    
    print_str = ''
    acc_over_data, loss_over_data = 0, 0
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

        inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)

        inputs_cross = F.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)

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

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs - num_bd - num_cross
        total_bd += num_bd
        total_cross += num_cross
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd + num_cross) :], dim=1) == total_targets[(num_bd + num_cross) :]
        )
        total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
        if num_cross:
            total_cross_correct += torch.sum(
                torch.argmax(total_preds[num_bd : (num_bd + num_cross)], dim=1)
                == total_targets[num_bd : (num_bd + num_cross)]
            )
            avg_acc_cross = total_cross_correct * 100.0 / total_cross

        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_bd = total_bd_correct * 100.0 / total_bd

        avg_loss_ce = total_loss_ce / total_sample
        
        all_bd_inputs.append(inputs_bd.detach().cpu())
        loss_over_data += loss.item()
        pred = total_preds.argmax(1, keepdim=True)
        acc_over_data += pred.eq(total_targets.view_as(pred)).sum().item()
        if verbose:
            print_str = 'Epoch: {}[{:3.1f}%] | tr_loss: {:.5f} | tr_acc: {:.2f}% | '.format(
                epoch, 100. * batch_idx / len(train_dl), 
                loss_over_data / min( (batch_idx+1) * train_dl.batch_size, len(train_dl.dataset) ), 
                100. * acc_over_data / min( (batch_idx+1) * train_dl.batch_size, len(train_dl.dataset) )
            )
            print('\r' + pre_str + update_color_of_str(print_str, color=color), end='')

    # print(f"Clean Accuracy, Clean: {avg_acc_clean}, Bd: {avg_acc_bd}, Cross: {avg_acc_cross}, epoch")

    schedulerC.step()
    n_samples = min( len(train_dl)*train_dl.batch_size, len(train_dl.dataset) )
    return loss_over_data/n_samples, acc_over_data/n_samples, update_color_of_str(print_str, color=color)


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    noise_grid,
    identity_grid,
    # best_clean_acc,
    # best_bd_acc,
    # best_cross_acc,
    # tf_writer,
    epoch,
    opt: Parameters,
    verbose: bool=False, pre_str: str='', color: str=None
):
    
    # print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()
    criterion_CE = torch.nn.CrossEntropyLoss()
    
    print_str = ''
    loss_over_data, acc_over_data = 0, 0
    all_bd_test_inputs = []
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
            
            preds_bd = netC(inputs_bd)
            loss_ce = criterion_CE(preds_bd, targets)
            loss = loss_ce

            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            # # Evaluate cross
            # if opt.cross_ratio:
            #     inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
            #     preds_cross = netC(inputs_cross)
            #     total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

            #     acc_cross = total_cross_correct * 100.0 / total_sample

            #     info_string = (
            #         "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | Cross: {:.4f}".format(
            #             acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc
            #         )
            #     )
            # else:
            #     info_string = "Clean Acc: {:.4f} - Bd Acc: {:.4f}".format(
            #         acc_clean, acc_bd
            #     )
            # progress_bar(batch_idx, len(test_dl), info_string)
        
        all_bd_test_inputs.append(inputs_bd.detach().cpu())    
        loss_over_data += loss.item()
        pred = preds_bd.argmax(1, keepdim=True)
        acc_over_data += pred.eq(targets_bd.view_as(pred)).sum().item()
        if verbose:
            print_str = '({:3.1f}%) ts_loss: {:.5f} | ts_acc: {:.2f}% | '
            print_str = print_str.format(
                100. * (batch_idx+1) / len(test_dl), 
                loss_over_data / min( (batch_idx+1) * test_dl.batch_size, len(test_dl.dataset) ), 
                100. * acc_over_data / min( (batch_idx+1) * test_dl.batch_size, len(test_dl.dataset) )
            )
            print('\r' + pre_str + update_color_of_str(print_str, color=color), end='')

    # print(f"Test Accuracy, Clean: {acc_clean}, Bd: {acc_bd}, epoch")
    n_samples = min( len(test_dl)*test_dl.batch_size, len(test_dl.dataset) )
    return loss_over_data/n_samples, acc_over_data/n_samples, update_color_of_str(print_str, color=color), torch.cat(all_bd_test_inputs, dim=0)


def main(torch_model: Torch_Model):
    
    opt = Parameters(torch_model)
    
    train_dl, test_dl = torch_model.data.prepare_data_loaders(batch_size=torch_model.model_configuration['batch_size'])
    
    netC, optimizerC = torch_model.model, torch_model.optimizer
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    # print("Train from scratch!!!")
    best_clean_acc = 0.0
    best_bd_acc = 0.0
    best_cross_acc = 0.0
    epoch_current = 0

    # Prepare grid
    ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        .to(opt.device)
    )
    array1d = torch.linspace(-1, 1, steps=opt.input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)

    for epoch in range(opt.n_iters):
        
        poisoned_test = train(
            netC, 
            optimizerC, 
            schedulerC, 
            train_dl, 
            noise_grid, 
            identity_grid, 
            # tf_writer, 
            epoch, 
            opt
        )
        
        best_clean_acc, best_bd_acc, best_cross_acc, bd_test_inputs = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl,
            noise_grid,
            identity_grid,
            best_clean_acc,
            best_bd_acc,
            best_cross_acc,
            # tf_writer,
            epoch,
            opt,
        )
        
        


if __name__ == "__main__":
    main()
    
    