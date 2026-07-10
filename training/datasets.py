import torch
import torchvision
import torchvision.transforms as T
import os

def get_mnist_loaders(data_dir='./data', batch_size=128, num_workers=0, image_size=32):
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))  # Maps [0, 1] to [-1, 1]
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def get_cifar10_loaders(data_dir='./data', batch_size=128, num_workers=0, image_size=32):
    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Maps [0, 1] to [-1, 1]
    ])
    
    test_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Maps [0, 1] to [-1, 1]
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def unnormalize(x):
    """Maps from [-1, 1] to [0, 1]"""
    return (x + 1.0) / 2.0
