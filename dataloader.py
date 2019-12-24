import os
import torch
import torch.utils.data as data
import torchvision.datasets as datasets


"""
PyTorch Custom Dataset
PyTorch supports two different types of datasets: Map-style and Iterable-style
A map-style dataset is one that implements the __getitem__() and __len__() protocols,
and represents a map from indices/keys to data samples.
An iterable-style dataset is an instance of a subclass of IterableDataset that implements the __iter__() protocol,
and represents an iterable over data samples. This type of datasets is particularly suitable for cases where random reads are expensive or even improbable,
and where the batch size depends on the fetched data.
For example, such a dataset, when called iter(dataset), could return a stream of data reading from a database,
a remote server, or even logs generated in real time.
"""
class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train', metafile = 'meta.txt',transform=None):
        """
		An example of map-style dataset
        args:
        root: path to data
        phase: train or val or test
        transform: if this is a dataset for vision problem, usually need a trandform function for data                              augmentation.
        """
        root = os.path.join(root, phase)
        # usually have a csv/txt file to store specific data path and label information
        metadata = os.path.join(root, metafile)
        self.data = utils.readfile(metadata)

    def __getitem__(self,idx):
        image = PILI.open(self.data[idx]['path']).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.data[idx]['label']
        return image, label

    def __len__(self):
        return len(self.data) 


# Standard dataloader
dataset = CustomDataset('./')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)


# How to fetch data
for batch_idx, batch_data in enumerate(dataloader):
	# do some thing...


# loading from a map-style dataset is roughly equivalent with:
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])


# Sampler
"""
torch.utils.data.Sampler classes are used to specify the sequence of indices/keys used in data loading.
They represent iterable objects over the indices to datasets.
E.g., in the common case with stochastic gradient decent (SGD),
a Sampler could randomly permute a list of indices and yield each one at a time,
or yield a small number of them for mini-batch SGD.
A custom Sampler that yields a list of batch indices at a time can be passed as the batch_sampler argument.
Automatic batching can also be enabled via batch_size and drop_last arguments.
An example, label sampler for few-shot learning episodic dataloader construction:
"""

class EpisodicSampler(data.Sampler):

    def __init__(self, total_classes, n_class, n_episode):
        self.total_classes = total_classes
        self.n_class = n_class
        self.n_episode = n_episode

    def __iter__(self):
        for i in range(self.n_episode):
            yield torch.randperm(self.total_classes)[:self.n_class]

    def __len__(self):
        return self.n_episode

