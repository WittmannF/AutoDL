from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def imshow(img, labels, classes):
    img = inverse_normalize(img, 
                            mean=IMGS_MEAN, 
                            std=IMGS_STD)
    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title([classes[l] for l in labels])
    plt.show()

def show_imgs_from_batch(img_loader, n_imgs, img_set):
    dataiter = iter(img_loader)
    images, labels = dataiter.next()
    images, labels = images[:n_imgs], labels[:n_imgs]
    plt.figure(figsize=(20, 20))
    imshow(torchvision.utils.make_grid(images, nrow=4), labels, img_set.classes)

def train_val_split(dataset, batch_size=16, validation_split=.2, 
                    shuffle_dataset=True, random_seed=42):
    """
    Helper code adapted from: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    """
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=batch_size, 
                                               sampler=train_sampler)
    
    validation_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader
