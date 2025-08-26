import torch
import numpy as np


# from .npca_custom_no_masking import NPCA_Custom
from .__deprecated__.v6.analyzer import PCA_Analyzer


class Purified_Net(torch.nn.Module):
    
    def __init__(self, analyzer: PCA_Analyzer):
        
        super().__init__()
        
        # self.npca = npca
        self.analyzer = analyzer
        
        return
    
    
    def detect(self, x, y):
        pc, score, label = self.analyzer.analyze(x.clone().cpu().numpy(), y)
        return label == self.analyzer.bad_label
    
    
    def forward(self, dataset):
        
        output = self.analyzer.model.model(dataset)
        output_random = torch.normal(0, 10, size=output.shape).to(output.device)
        
        output_classes = output.argmax(1).cpu().numpy().reshape(-1)
        
        positives = self.detect(dataset, output_classes)
        vulnerable_ind = (output_classes==self.analyzer.target_class)
        vulnerable_ind = vulnerable_ind & positives
        
        output_random = torch.normal(0, 10, size=output.shape).to(output.device)
        output[vulnerable_ind] = output_random[vulnerable_ind].clone()
        
        return output
    
    