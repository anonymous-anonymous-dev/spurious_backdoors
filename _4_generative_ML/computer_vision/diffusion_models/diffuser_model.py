


class Diffuser_Model:
    
    def __init__(
        self, 
        model_configuration: dict={},
        results_path: str='', 
        model_name: str='',
        verbose: bool=True, 
        **kwargs
    ):
        
        self.model_configuration = {
            'batch_size': 64
        }
        for key in model_configuration.keys():
            self.model_configuration[key] = model_configuration[key]
        self.batch_size = self.model_configuration['batch_size']
        
        self.model_name = results_path + model_name
            
        self.last_hidden_state_dim = None
        self.verbose = verbose
        
        return
    
    
    def not_implemented(self, *args, **kwargs): 
        raise NotImplementedError('This is the parent class. Please call an instance of child class to use this functionality.')
    
    
    def prepare_model(self): return self.not_implemented()
        