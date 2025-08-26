import torch, torchvision
import numpy as np



class Torch_Dataset:
    
    def __init__(
        self, 
        data_name: str, 
        preferred_size: int=0, 
        data_means: list[int]=[0], data_stds: list[int]=[1]
    ):
        
        self.data_name = data_name
        self.preferred_size = preferred_size
        if isinstance(self.preferred_size, int):
            self.preferred_size = (self.preferred_size, self.preferred_size)
        self.channels = None
        
        self.data_means = data_means
        self.data_stds = data_stds
        
        self.train = None; self.default_train_transform = None
        self.test = None; self.default_test_transform = None
        self.num_classes = None
        
        return
    
    
    def renew_data(self): self.not_implemented()
    def not_implemented(self):
        print_str = 'This is the parent class. Please call the corresponding function '
        print_str += 'of the specific dataset to get the functionality.'
        assert False, print_str
        return
    def get_output_shape(self): return tuple([len(self.get_class_names())])
    def get_input_shape(self): return self.train.__getitem__(0)[0].shape
    def get_class_names(self) -> list:  return np.arange(len(np.unique( [self.train[i][1] for i in range(self.train.__len__())] )))
    def compute_num_classes(self): return 1+np.max([self.train[i][1] for i in range(self.train.__len__())])
    def get_num_classes(self): 
        if self.num_classes is None:
            self.num_classes = self.compute_num_classes()
        return self.num_classes
    
    
    def reset_transforms(self, *args, **kwargs):
        self.train.transform = self.default_train_transform
        self.test.transform = self.default_test_transform
        return
    
    
    def update_transforms(self, transform: torchvision.transforms, subdata_category: str='all'):
        
        if subdata_category == 'all':
            self.train.transform = transform
            self.test.transform = transform
        elif subdata_category == 'train':
            self.train.transform = transform
        elif subdata_category == 'test':
            self.test.transform = transform
            
        return
    
    
    def prepare_data_loaders(self, batch_size=64, shuffle: bool=True):
        
        self.batch_size = batch_size
        
        train_loader = torch.utils.data.DataLoader(self.train, shuffle=shuffle, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(self.test, shuffle=shuffle, batch_size=batch_size)
        
        return train_loader, test_loader
    
    
    def sample_data(self, data_type, num_samples: int=100, batch_size: int=64, shuffle: bool=False):
        
        # make a data loader
        data_loader = torch.utils.data.DataLoader(data_type, shuffle=shuffle, batch_size=batch_size)
        
        data_x, data_y = [], []
        for i, (x, y) in enumerate(data_loader):
            print(f'\rSampling data: {len(data_x)}/{num_samples}', end='')
            data_x = torch.cat([data_x, x], 0) if len(data_x)>0 else x
            data_y = torch.cat([data_y, y], 0) if len(data_y)>0 else y
            if len(data_x) >= num_samples:
                return (data_x, data_y)
        
        # sample_size = data_type.__len__()
        # if num_samples:
        #     sample_size = min(sample_size, num_samples)
        
        # data_indices = np.random.choice(data_type.__len__(), sample_size, replace=False)
        # data_x, data_y = [], []
        # for i, ind in enumerate(data_indices):
        #     print(f'\rSampling data: {i+1}/{len(data_indices)}', end='')
        #     x, y = data_type.__getitem__(ind)
        #     data_x.append(x)
        #     data_y.append(y)
        # data_x, data_y = torch.stack(data_x, dim=0), torch.tensor(data_y)
        
        return (data_x, data_y)
    
    
    def sample_data_of_certain_class(self, data_type, batch_size=64, class_: int=0):
        
        data_loader = torch.utils.data.DataLoader(data_type, batch_size=5000, shuffle=False)
        
        sample_size = data_type.__len__()
        if batch_size:
            sample_size = min(sample_size, batch_size)
            
        data_x, data_y = [], []
        for k, (input, target) in enumerate(data_loader):
            print(f'\rSampling data: {len(data_x)}/{sample_size}', end='')
            if len(data_x)==0:
                data_x = input[target == class_]
                data_y = target[target == class_]
            else:
                data_x = torch.cat( [data_x, input[target == class_]], 0 )
                data_y = torch.cat( [data_y, target[target == class_]], 0 )
                
            if len(data_x) > sample_size:
                return (data_x, data_y)
        
        return (data_x, data_y)
    
    
    def view_random_samples(self, n_rows, n_cols):
        
        import matplotlib.pyplot as plt
        
        random_indices = np.random.choice(self.train.__len__(), size=n_rows*n_cols).reshape(n_rows, n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_rows, n_cols))
        for n_r in range(n_rows):
            for n_c in range(n_cols):
                ax = axs[n_r][n_c]
                
                x, y = self.train.__getitem__(random_indices[n_r, n_c])
                
                ax.imshow( np.transpose(x, (1,2,0)) )
                ax.set_title( y )
                ax.set_xticks([])
                ax.set_yticks([])
                
        plt.tight_layout()
        
        return
    
    