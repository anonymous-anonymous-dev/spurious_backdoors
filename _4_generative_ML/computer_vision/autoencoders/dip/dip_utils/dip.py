import torch
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr


from .._dip_library.models import *
from .._dip_library.utils.denoising_utils import *



dtype = torch.cuda.FloatTensor


class DIP:
    
    def __init__(
        self,
        input_depth=32, reg_noise_std=1/30,
        exp_weight=0.99, 
        noise_size=128,
        **kwargs
    ):
        
        self.input_depth = input_depth
        self.reg_noise_std = reg_noise_std

        self.exp_weight = exp_weight
        self.out_avg = None
        self.last_net = None
        
        self.noise_size = noise_size
        
        self.flag = 0
        
        self.loss_type='conventional'
        self.loss_values = []
        
        self.dtype = dtype
        self.learning_rate = 1e-2
        
        # self.prepare()
        
        return
    
    
    def prepare(self, net_input, net_output):
        
        self.net_input, self.net_output = net_input, net_output
        
        # Model
        pad = 'reflection'
        self.net = get_net(
            self.input_depth, 'skip', pad,
            skip_n33d=128,
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            upsample_mode='bilinear'
        ).type(dtype)
        
        # Loss and Optimizer
        self.mse = torch.nn.MSELoss().type(dtype)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # self.update_evaluation_metrics(initialize=True)
        self.last_net = [x.detach().cpu() for x in self.net.parameters()]
        # self.actual_loss_last = self.run_metrics[list(self.run_metrics.keys())[0]][0]
        print(self.net_input.shape)
        
        self.out_avg = torch.zeros_like(self.net(self.net_input)).detach()
        
        return
    
    
    def predict(self, input=None):
        return self.net(self.net_input) if input is None else self.net(input)
    def loss_fn(self, out_pt, out_gt): return torch.mean(torch.square(out_gt-out_pt))
    
    
    def choose_best_net(self):
        
        if self.loss_values[-1] - self.actual_loss_last > 0:
            for new_param, net_param in zip(self.last_net, self.net.parameters()):
                net_param.data.copy_(new_param.cuda())
            self.flag += 1
        else:
            self.last_net = [x.detach().cpu() for x in self.net.parameters()]
            self.actual_loss_last = self.loss_values[-1]
            self.flag = 0
            
        return
    
    
    def step(self, net_input, net_output):
        
        if self.reg_noise_std > 0:
            net_input = net_input.detach().clone().normal_() * self.reg_noise_std
            net_input += net_input.detach().clone()
        
        self.optimizer.zero_grad()
        
        # Perform Inference
        out = self.predict(net_input)
        # Smoothing
        # self.out_avg = self.exp_weight*self.out_avg + (1 - self.exp_weight)*out.detach()
        
        total_loss = self.loss_fn(out, net_output)
        total_loss.backward()
        self.optimizer.step()
        
        self.loss_values.append(total_loss.item())
        
        return total_loss.item()
    
    
    def run(self, num_iter=5000, revise_iterations=0):
        
        for i in range(num_iter):
            
            self.step(self.net_input, self.net_output)
            
            if revise_iterations > 0:
                if i % revise_iterations == 0:
                    self.choose_best_net()
        
        return
    
    