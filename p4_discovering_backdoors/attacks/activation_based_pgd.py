import torch
import numpy as np
from tqdm import tqdm


from .._imported_implementations_.spurious_imagenet.neural_pca.activations import activations_with_grad
from .._imported_implementations_.spurious_imagenet.neural_pca.adversarial_attacks.act_apgd import ActivationAPGDAttack



class Activation_Based_PGD(ActivationAPGDAttack):
    
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        return
    
    
    def deprecated_loss_obj_full(self, alpha, y):
        alpha = alpha * torch.sum(self.eigenvecs, dim=0, keepdim=True)
        return alpha[np.arange(len(y)),y] 
    
    
    def attack_single_run(self, x, y, x_init=None, first_run=False):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        all_losses = []

        if x_init is not None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
        elif first_run:
            print('First run, intialized with an image.')
            x_adv = x.clone()
        else:
            print('Standard normalization is being used norm is', self.norm)
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x).detach() * self.normalize(t) * self.masks
            
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
                                 ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
                                      ).to(self.device)
        assert not self.is_tf_model
        
        
        # #############################
        # Step 1: Compute Loss and Get Gradients
        # #############################
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                activations = activations_with_grad(x_adv, self.model, self.last_layer, grad_enabled=True)
                weighted_act = activations * self.last_layer.weight[self.target_cls]

                pca_activations = weighted_act@self.eigenvecs
                loss_indiv = self.criterion_indiv(pca_activations, y)

                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        
        
        # #############################
        # Step 2: Initialize best loss and put in the list
        # #############################
        loss_best = loss_indiv.detach().clone()
        if self.return_all_losses_confid:
            #activations = self.ll_activations.attribute(x, attribute_to_layer_input=True)
            activations = activations_with_grad(x, self.model, self.last_layer)
            weighted_act = activations * self.last_layer.weight[self.target_cls]

            pca_activations = weighted_act@self.eigenvecs 
            loss_indiv_ = self.criterion_indiv(pca_activations, y)
            
            all_losses.append(loss_indiv_.detach().cpu().unsqueeze(dim=1))
            print('all losses', all_losses)

        print('loss indiv shape is', loss_indiv.shape)
        
        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *([1] * self.ndims)]).to(self.device).detach()

        x_adv_old = x_adv.clone()
        k = self.n_iter_2 + 0
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        # #############################
        # Step 3: Compute Loss and Get Gradients
        # #############################
        n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]
        u = torch.arange(x.shape[0], device=self.device)
        for i in tqdm(range(self.n_iter)):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                x_adv_1 = x_adv + step_size * self.masks * self.normalize(grad)
                x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x) * torch.min(self.eps * torch.ones_like(x).detach(), self.lp_norm(x_adv_1 - x)), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x) * torch.min(self.eps * torch.ones_like(x).detach(), self.lp_norm(x_adv_1 - x)), 0.0, 1.0)

                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    activations = activations_with_grad(x_adv, self.model, self.last_layer, grad_enabled=True)
                    weighted_act = activations * self.last_layer.weight[self.target_cls]
                    
                    pca_activations = weighted_act@self.eigenvecs
                    loss_indiv = self.criterion_indiv(pca_activations, y)

                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            grad /= float(self.eot_iter)

            ### check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()

                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(loss_steps, i, k,
                                                            loss_best, k3=self.thr_decr)
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                            loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation,
                                                fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()
                    # print('prev k is', k)
                    k = max(k - self.size_decr, self.n_iter_min)
                    # print('next k is', k)

                    counter3 = 0

            if self.return_all_losses_confid:
                activations = activations_with_grad(x_adv, self.model, self.last_layer)
                weighted_act = activations * self.last_layer.weight[self.target_cls]
                
                pca_activations = weighted_act@self.eigenvecs
                loss_indiv_ = self.criterion_indiv(pca_activations, y)
                
                all_losses.append(loss_indiv_.cpu().unsqueeze(dim=1))
                if self.verbose:
                    print('losses:', all_losses[-1])
                    print('pca_activations:', pca_activations)

        if self.return_all_losses_confid:
            return (x_best, loss_best, x_best_adv, torch.cat(all_losses, dim=1))
        else:
            return (x_best, loss_best, x_best_adv)
    

