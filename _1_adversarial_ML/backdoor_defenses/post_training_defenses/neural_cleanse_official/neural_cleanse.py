import torch
import numpy as np


from _0_general_ML.model_utils.torch_model import Torch_Model

from .neural_cleanse_helpers import Recorder, RegressionModel


class Neural_Cleanse:
    """
    Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks
    URL: https://ieeexplore.ieee.org/abstract/document/8835365/
    
    @inproceedings{wang2019neural,
        title={Neural cleanse: Identifying and mitigating backdoor attacks in neural networks},
        author={Wang, Bolun and Yao, Yuanshun and Shan, Shawn and Li, Huiying and Viswanath, Bimal and Zheng, Haitao and Zhao, Ben Y},
        booktitle={2019 IEEE symposium on security and privacy (SP)},
        pages={707--723},
        year={2019},
        organization={IEEE}
    }
    
    CREDITS: This code was taken and modified from: https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master
    """
    
    def __init__(self, torch_model: Torch_Model, configuration: dict={}, data_means: list[int]=[0], data_stds: list[int]=[1]):
        
        self.configuration = {
            'learning_rate': 1e-1,
            'epochs': 50,
            'epsilon': 1e-7,
            'init_cost': 1e-3,
            'cost_multiplier': 2,
            'n_times_test': 5,
            'atk_succ_threshold': 99
        }
        for key in configuration.keys():
            self.configuration[key] = configuration[key]
        
        self.torch_model = torch_model
        self.data_means = data_means
        self.data_stds = data_stds
        
        return
    
    
    def train_step(
        self,
        regression_model: RegressionModel, 
        optimizerR, 
        dataloader: torch.utils.data.DataLoader, 
        recorder: Recorder, 
        epoch, 
        target_label_in, 
        atk_succ_threshold: float=99,
        early_stop: bool=True,
        early_stop_threshold: float=99,
        early_stop_patience: float=99,
        patience: int=5,
        init_cost=1e-3,
        **kwargs
        # opt
    ):
        
        torch_device = regression_model.torch_model.device
        
        print("Epoch {} - Label: {}:".format(epoch, target_label_in))
        # Set losses
        cross_entropy = torch.nn.CrossEntropyLoss()
        total_pred = 0
        true_pred = 0

        # Record loss for all mini-batches
        loss_ce_list = []
        loss_reg_list = []
        loss_list = []
        loss_acc_list = []

        # Set inner early stop flag
        inner_early_stop_flag = False
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Forwarding and update model
            optimizerR.zero_grad()

            inputs = inputs.to(regression_model.torch_model.device)
            sample_num = inputs.shape[0]
            total_pred += sample_num
            target_labels = torch.ones((sample_num), dtype=torch.int64).to(torch_device) * target_label_in
            predictions = regression_model(inputs)

            loss_ce = cross_entropy(predictions, target_labels)
            loss_reg = torch.norm(regression_model.get_raw_mask(), 2)
            total_loss = loss_ce + recorder.cost * loss_reg
            total_loss.backward()
            optimizerR.step()

            # Record minibatch information to list
            minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
            loss_ce_list.append(loss_ce.detach())
            loss_reg_list.append(loss_reg.detach())
            loss_list.append(total_loss.detach())
            loss_acc_list.append(minibatch_accuracy)

            true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()
            # progress_bar(batch_idx, len(dataloader))

        loss_ce_list = torch.stack(loss_ce_list)
        loss_reg_list = torch.stack(loss_reg_list)
        loss_list = torch.stack(loss_list)
        loss_acc_list = torch.stack(loss_acc_list)

        avg_loss_ce = torch.mean(loss_ce_list)
        avg_loss_reg = torch.mean(loss_reg_list)
        avg_loss = torch.mean(loss_list)
        avg_loss_acc = torch.mean(loss_acc_list)

        # Check to save best mask or not
        if avg_loss_acc >= atk_succ_threshold and avg_loss_reg < recorder.reg_best:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()
            recorder.reg_best = avg_loss_reg
            # recorder.save_result_to_dir(opt)
            print(" Updated !!!")

        # Show information
        print(
            "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
                true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
            )
        )

        # Check early stop
        if early_stop:
            if recorder.reg_best < float("inf"):
                if recorder.reg_best >= early_stop_threshold * recorder.early_stop_reg_best:
                    recorder.early_stop_counter += 1
                else:
                    recorder.early_stop_counter = 0

            recorder.early_stop_reg_best = min(recorder.early_stop_reg_best, recorder.reg_best)

            if (
                recorder.cost_down_flag
                and recorder.cost_up_flag
                and recorder.early_stop_counter >= early_stop_patience
            ):
                print("Early_stop !!!")
                inner_early_stop_flag = True

        if not inner_early_stop_flag:
            # Check cost modification
            if recorder.cost == 0 and avg_loss_acc >= atk_succ_threshold:
                recorder.cost_set_counter += 1
                if recorder.cost_set_counter >= patience:
                    recorder.reset_state(init_cost)
            else:
                recorder.cost_set_counter = 0

            if avg_loss_acc >= atk_succ_threshold:
                recorder.cost_up_counter += 1
                recorder.cost_down_counter = 0
            else:
                recorder.cost_up_counter = 0
                recorder.cost_down_counter += 1

            if recorder.cost_up_counter >= patience:
                recorder.cost_up_counter = 0
                print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
                recorder.cost *= recorder.cost_multiplier_up
                recorder.cost_up_flag = True

            elif recorder.cost_down_counter >= patience:
                recorder.cost_down_counter = 0
                print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
                recorder.cost /= recorder.cost_multiplier_down
                recorder.cost_down_flag = True

            # Save the final version
            if recorder.mask_best is None:
                recorder.mask_best = regression_model.get_raw_mask().detach()
                recorder.pattern_best = regression_model.get_raw_pattern().detach()

        return inner_early_stop_flag
    
    
    def train(
        self,
        # opt, 
        test_dataloader, 
        init_mask, init_pattern, target_label_in
    ) -> Recorder:

        # test_dataloader = torch.utils.data.DataLoader(data.test)
        
        # Build regression model
        regression_model = RegressionModel(self.torch_model, init_mask, init_pattern, data_means=self.data_means, data_stds=self.data_stds).to(self.torch_model.device)

        # Set optimizer
        optimizerR = torch.optim.Adam(regression_model.parameters(), lr=self.configuration['learning_rate'], betas=(0.5, 0.9))

        # Set recorder (for recording best result)
        recorder = Recorder(**self.configuration)

        for epoch in range(self.configuration['epochs']):
            early_stop = self.train_step(
                regression_model, 
                optimizerR, 
                test_dataloader, 
                recorder, 
                epoch, 
                target_label_in,
                **self.configuration
                # opt
            )
            if early_stop:
                break

        return recorder
    
    
    def outlier_detection(self, l1_norm_list, idx_mapping):
        print("-" * 30)
        print("Determining whether model is backdoor")
        consistency_constant = 1.4826
        median = torch.median(l1_norm_list)
        mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
        # ------------------------------
        # personally added zero correction
        mad += self.configuration['epsilon'] if mad==0 else 0
        # ------------------------------
        min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

        print(f"Median: {median}, MAD: {mad}, Anomaly index: {min_mad}")
        if min_mad < 2:
            print("Not a backdoor model")
        else:
            print("This is a backdoor model")
            
        flag_list = []
        for y_label in idx_mapping:
            if l1_norm_list[idx_mapping[y_label]] > median:
                continue
            if (torch.abs(l1_norm_list[idx_mapping[y_label]]-median)/mad) > 2:
                flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))
                
        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            
        print(f"Flagged label list: {",".join([f"{y_label}: {l_norm}" for y_label, l_norm in flag_list])}")
        
        return flag_list
    
    
    def analyze(self, test_dataset: torch.utils.data.Dataset, target_labels: list[int]=[0, 1]):
        
        from utils_.torch_utils import get_data_samples_from_loader
        
        x = test_dataset.__getitem__(0)[0]
        init_mask = np.ones_like(x[:1]).astype(np.float32)
        init_pattern = np.ones_like(x).astype(np.float32)
        
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.torch_model.model_configuration['batch_size'], shuffle=False)
        
        for test in range(self.configuration['n_times_test']):
            print("Test {}:".format(test))
            
            masks = []
            idx_mapping = {}
            for target_label in target_labels:
                print("----------------- Analyzing label: {} -----------------".format(target_label))
                # opt.target_label = target_label
                recorder = self.train(test_dataloader, init_mask, init_pattern, target_label)
                
                mask = recorder.mask_best
                masks.append(mask)
                idx_mapping[target_label] = len(masks) - 1

            l1_norm_list = torch.stack([torch.sum(torch.abs(m)) for m in masks])
            print("{} labels found".format(len(l1_norm_list)))
            print("Norm values: {}".format(l1_norm_list))
            flag_list = self.outlier_detection(l1_norm_list, idx_mapping)
            
        return flag_list
    
    