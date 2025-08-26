import torch



class Quantization(torch.nn.Module):
    
    def __init__(self, quantization_levels=16, quantization_hardness=1, quantization_thresholds: list[int]=None, **kwargs):
        
        super().__init__()
        
        self.quantization_hardness = quantization_hardness
        
        self.quantization_levels = quantization_levels
        self.quantization_thresholds_interval = 1 / quantization_levels
        
        self.quantization_thresholds = [ (i+1)*self.quantization_thresholds_interval for i in range(quantization_levels)]
        if quantization_thresholds is not None:
            if len(quantization_thresholds) == quantization_levels:
                self.quantization_thresholds = quantization_thresholds
        
        return
        
    def sigmoid_fn(self, x_in):
        return 1/(1+torch.exp(-x_in))
        
        
    def __forward(self, inputs):
        
        # min_ = min(0, torch.min(inputs))
        # max_ = max(1, torch.max(inputs))
        min_ = torch.min(inputs)
        max_ = torch.max(inputs)
        if max_-min_ == 0:
            max_=1; min_=0
        
        inputs = (inputs-min_)/(max_-min_)
        
        output = 0 * inputs
        for quantization_threshold in self.quantization_thresholds:
            output += self.sigmoid_fn(
                self.quantization_hardness * (inputs - quantization_threshold)
            )
        
        output /= (self.quantization_levels - 1)
        
        return torch.clamp(output, 0, 1) * (max_-min_) + min_
    
    
    def forward(self, inputs):
        
        min_ = torch.min(inputs)
        max_ = torch.max(inputs)
        if max_-min_ == 0:
            max_=1; min_=0
        
        inputs = (inputs-min_)/(max_-min_)
        
        output = torch.round((inputs+1) * self.quantization_levels)/self.quantization_levels
        output -= 1.
        
        return torch.clamp(output, 0, 1) * (max_-min_) + min_
    
    