import numpy as np
import torch
import torchvision
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

np.random.seed(546)

class CIFAR10Dataset(torchdata.Dataset):

    def __init__(self, train=True):
        self.transform = transforms.Compose(
                            [transforms.ToTensor(),])
        self.dataset = torchvision.datasets.CIFAR10('./', train=train, download=True, transform=self.transform)
        self.text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.image_shape = (3, 32, 32)
        self.n_classes = 10
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = 2*img - 1.
        text_label = self.text_labels[label]
        return img, label, text_label

    def __len__(self):
        return len(self.dataset)

d = CIFAR10Dataset()

for i in range(1):
    i = np.random.randint(0, len(d))
    img, class_label, text = d[i]
    img = (np.transpose(img.reshape((3, 32, 32)), (1, 2, 0))+1)/2
    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    plt.title(text)
    plt.show()