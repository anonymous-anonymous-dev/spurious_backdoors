# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt

# OS
import os
import argparse

from p4_discovering_backdoors.model_utils.torch_model_save_best import Torch_Model_Save_Best

from p4_discovering_backdoors.helper.data_helper import prepare_clean_and_poisoned_data
from p4_discovering_backdoors.helper.helper_class import Helper_Class

from p4_discovering_backdoors.config import *

from _0_general_ML.data_utils.torch_dataset import Torch_Dataset as My_Torch_Dataset
# from __sota_forks__.BackdoorBench.attack.lc import LabelConsistent as LabelConsistent_SOTA

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # # Input size: [batch, 3, 32, 32]
        # # Output size: [batch, 3, 32, 32]
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
        #     nn.ReLU(),
        #     nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
        #     nn.ReLU(),
		# 	nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
        #     nn.ReLU(),
		# 	# nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
        #     # nn.ReLU(),
        # )
        # self.decoder = nn.Sequential(
        #     # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
        #     # nn.ReLU(),
		# 	nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
        #     nn.ReLU(),
		# 	nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
        #     nn.Sigmoid(),
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),           # [batch, 24, 8, 8]
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
		    nn.ConvTranspose2d(48, 32, 3, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),   # [batch, 12, 16, 16]
            # nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    
def main(valid: bool=True):
    
    # parser = argparse.ArgumentParser(description="Train Autoencoder")
    # parser.add_argument("--valid", action="store_true", default=True, help="Perform validation only.")
    # args = parser.parse_args()
    
    # Create model
    autoencoder = create_model()

    # Load data
    # *** setting up the experiment ***
    experiment_folder = 'results_4/'
    dataset_name = 'cifar10'
    backdoor_attack_type = 'simple_backdoor'
    poisoning_ratio = 0.3

    # *** preparing some results-related variables ***
    results_path = '../../__all_results__/_p4_discovering_backdoors/' + experiment_folder
    my_model_configuration = model_configurations[dataset_name]
    my_model_configuration['dataset_name'] = dataset_name
    csv_file_path = results_path + my_model_configuration['dataset_name'] + '/csv_file/'

    my_backdoor_configuration = {
        'poison_ratio': poisoning_ratio, 
        'poison_ratio_wrt_class_members': True,
        'type': backdoor_attack_type
    }
    # if backdoor_attack_type == 'clean_label_backdoor':
    #     my_backdoor_configuration['epsilon'] = 0.1

    helper = Helper_Class(my_model_configuration=my_model_configuration, my_backdoor_configuration=my_backdoor_configuration)
    helper.prepare_paths_and_names(results_path, csv_file_path, model_name_prefix='central', filename='accuracies_and_losses_test.csv')

    my_data, poisoned_data = prepare_clean_and_poisoned_data(my_model_configuration, my_backdoor_configuration)
    helper.check_conducted(data_name=my_data.data_name, count_continued_as_conducted=False)
    
    trainloader, testloader = my_data.prepare_data_loaders(batch_size=my_model_configuration['batch_size'])
    classes = my_data.get_class_names()

    if valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load("./_1_adversarial_ML/backdoor_attacks/label_consistent_helpers/autoencoder.pkl"))
        dataiter = iter(testloader)
        
        images, labels = next(dataiter)
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        
        from utils_.general_utils import normalize
        from utils_.visual_utils import show_image_grid
        
        images = images[:20].detach().cpu().numpy()
        decoded_imgs = autoencoder(Variable(torch.tensor(images).cuda()))[1].detach().cpu().numpy()
        
        decoded_imgs = decoded_imgs - np.min(decoded_imgs)
        decoded_imgs /= np.max(decoded_imgs)
        
        show_image_grid(normalize(images), channels_first=True, n_rows=2, n_cols=10)
        show_image_grid(normalize(decoded_imgs), channels_first=True, n_rows=2, n_cols=10)
        # imshow(torchvision.utils.make_grid(images[:20]))
        # imshow(torchvision.utils.make_grid(decoded_imgs[:20].data))
        
    else:
        pass;
        # # Define an optimizer and criterion
        # criterion = nn.BCELoss()
        # optimizer = optim.Adam(autoencoder.parameters())

        # for epoch in range(100):
        #     running_loss = 0.0
        #     for i, (inputs, _) in enumerate(trainloader, 0):
        #         inputs = get_torch_vars(inputs)

        #         # ============ Forward ============
        #         encoded, outputs = autoencoder(inputs)
        #         loss = criterion(outputs, inputs)
        #         # ============ Backward ============
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #         # ============ Logging ============
        #         running_loss += loss.data
        #         if i % 2000 == 1999:
        #             print('[%d, %5d] loss: %.3f' %
        #                 (epoch + 1, i + 1, running_loss / 2000))
        #             running_loss = 0.0

        # print('Finished Training')
        # print('Saving Model...')
        # if not os.path.exists('./weights'):
        #     os.mkdir('./weights')
        # torch.save(autoencoder.state_dict(), ".autoencoder.pkl")


# if __name__ == '__main__':
#     main()