from torchvision import transforms
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler, Spiral_Handler, Quadratic_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10, get_Spiral, get_Quadratic
from nets import Net, MNIST_Net, SVHN_Net, CIFAR10_Net, Spiral_Net, Quadratic_Net
from query_strategies import EntropySampling, KMeansSampling, BALDDropout, AdversarialBIM, RandomSampling, LeastConfidence

# MarginSampling, LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, KMeansSampling, KCenterGreedy, BALDDropout, AdversarialBIM, AdversarialDeepFool

params = {'MNIST':
              {'n_epoch': 10, 
               'train_args':{'batch_size': 64, 'num_workers': 1},
               'test_args':{'batch_size': 1000, 'num_workers': 1},
               'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
          'FashionMNIST':
              {'n_epoch': 10, 
               'train_args':{'batch_size': 64, 'num_workers': 1},
               'test_args':{'batch_size': 1000, 'num_workers': 1},
               'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
          'SVHN':
              {'n_epoch': 20, 
               'train_args':{'batch_size': 64, 'num_workers': 1},
               'test_args':{'batch_size': 1000, 'num_workers': 1},
               'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
          'CIFAR10':
              {'n_epoch': 20, 
               'train_args':{'batch_size': 64, 'num_workers': 1},
               'test_args':{'batch_size': 1000, 'num_workers': 1},
               'optimizer_args':{'lr': 0.05, 'momentum': 0.3}},
          'Spiral':
              {'n_epoch': 2000,
               'train_args':{'batch_size': 16, 'num_workers': 0}, # 40 train + 10 test
               'test_args':{'batch_size': 10, 'num_workers': 0},
               'optimizer_args':{'lr': 0.001, 'momentum': 0.9, 'weight_decay': 3e-3}},
          'Blob':
              {'n_epoch': 20,
               'train_args':{'batch_size': 16, 'num_workers': 0}, # 80 train + 20 test
               'test_args':{'batch_size': 10, 'num_workers': 0},
               'optimizer_args':{'lr': 0.01, 'momentum': 0.9}},
          'Quadratic':
              {'n_epoch': 2000,
               'train_args':{'batch_size': 16, 'num_workers': 0}, # 80 train + 20 test
               'test_args':{'batch_size': 10, 'num_workers': 0},
               'optimizer_args':{'lr': 0.001, 'momentum': 0.9, 'weight_decay': 3e-3}
          
          }}

def get_handler(name):
    if name == 'MNIST':
        return MNIST_Handler
    elif name == 'FashionMNIST':
        return MNIST_Handler
    elif name == 'SVHN':
        return SVHN_Handler
    elif name == 'CIFAR10':
        return CIFAR10_Handler
    elif name == 'Spiral':
        return Spiral_Handler
    elif name == 'Blob':
        return Blob_Handler
    elif name == 'Quadratic':
        return Quadratic_Handler

def get_dataset(name):
    if name == 'MNIST':
        return get_MNIST(get_handler(name))
    elif name == 'FashionMNIST':
        return get_FashionMNIST(get_handler(name))
    elif name == 'SVHN':
        return get_SVHN(get_handler(name))
    elif name == 'CIFAR10':
        return get_CIFAR10(get_handler(name))
    elif name == 'Spiral':
        return get_Spiral(get_handler(name))
    elif name == 'Blob':
        return get_Blob(get_handler(name))
    elif name == 'Quadratic':
        return get_Quadratic(get_handler(name))
    else:
        raise NotImplementedError
        
def get_net(name, device):
    if name == 'MNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'FashionMNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'SVHN':
        return Net(SVHN_Net, params[name], device)
    elif name == 'CIFAR10':
        return Net(CIFAR10_Net, params[name], device)
    elif name == 'Spiral':
        return Net(Spiral_Net, params[name], device)
    elif name == 'Blob':
        return Net(Blob_Net, params[name], device)
    elif name == 'Quadratic':
        return Net(Quadratic_Net, params[name], device)    
    else:
        raise NotImplementedError
    
def get_params(name):
    return params[name]

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    #elif name == "MarginSampling":
        #return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    #elif name == "LeastConfidenceDropout":
        #return LeastConfidenceDropout
    #elif name == "MarginSamplingDropout":
        #return MarginSamplingDropout
    #elif name == "EntropySamplingDropout":
        #return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    #elif name == "KCenterGreedy":
        #return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    #elif name == "AdversarialDeepFool":
        #return AdversarialDeepFool
    else:
        raise NotImplementedError
