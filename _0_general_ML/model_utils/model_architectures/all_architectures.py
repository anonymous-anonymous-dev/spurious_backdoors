from .mnist import MNIST_CNN, Conv2D

from .gtsrb import Resnet18_GTSRB, Resnet50_GTSRB, CNN_GTSRB

from .cifar10 import vgg11 as cifar10_vgg11
from .cifar10_resnet import Resnet18_CIFAR10
from .cifar10_vit import ViT16_CIFAR10 as ViT16_CIFAR10_Official
# from .vit_custom import ViT16_CIFAR10, ViT16_CIFAR100
# from .vit_custom_v2 import ViT16_CIFAR10, ViT16_CIFAR100
from .vit_custom_v3 import ViT16_CIFAR10, ViT16_CIFAR100

from .cifar4 import vgg11 as cifar4_vgg11
from .cifar10_resnet50 import Resnet50_CIFAR10

# from .cifar100_resnets import Resnet50_CIFAR100, Resnet18_CIFAR100
from .cifar100_resnets_official import resnet50 as Resnet50_CIFAR100, resnet18 as Resnet18_CIFAR100, resnet18_classes10 as Resnet18_Custom_CIFAR10
from .cifar100_resnets_official import resnet18_for_gtsrb

from .imagenet_resnet50 import Resnet50_Imagenet, Resnet18_Imagenet, ViT_B_Imagenet
from .vit_imagenet import ViT16_Imagenet
