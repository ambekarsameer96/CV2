from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Helper function to quickly see the values of a list or dictionary of data
def printTensorList(data, detailed=False):
    if isinstance(data, dict):
        print('Dictionary Containing: ')
        print('{')
        for key, tensor in data.items():
            print('\t', key, end='')
            print(' with Tensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print('}')
    else:
        print('List Containing: ')
        print('[')
        for tensor in data:
            print('\tTensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print(']')


class SwappedDatasetLoader(Dataset):

    def __init__(self, data_file, prefix, resize=256):
        self.prefix = prefix
        self.resize = resize
        # Define your initializations and the transforms here. You can also
        # define your tensor transforms to normalize and resize your tensors.
        # As a rule of thumb, put anything that remains constant here.
        self.lines = open(data_file, 'r')
        self.data_paths = []
        for i, line in enumerate(self.lines.readlines()):
            file_dict = {}

            line_clean = line.rstrip()
            line_parts = line_clean.split("_")

            file_dict['sw'] = line_clean
            file_dict['mask'] = line_clean.replace('sw', 'mask')
            file_dict['fg'] = line_parts[0] + '_fg_' + line_parts[-1]
            file_dict['bg'] = line_parts[0] + '_bg_' + line_parts[2] + '.png'

            self.data_paths.append(file_dict)

    def get_img(self, path):
        return transforms.ToTensor()(Image.open(path))

    def __len__(self):
        # Return the length of the datastructure that is your dataset
        return len(self.data_paths)

    def __getitem__(self, index):
        source = self.get_img(self.prefix + self.data_paths[index]['sw'])
        target = self.get_img(self.prefix + self.data_paths[index]['bg'])
        swap = self.get_img(self.prefix + self.data_paths[index]['fg'])
        mask = self.get_img(self.prefix + self.data_paths[index]['mask'])
        image_dict = {'source': source,
                      'target': target,
                      'swap': swap,
                      'mask': mask}

        return image_dict, self.data_paths[index]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time
    # It is always a good practice to have separate debug section for your
    # functions. Test if your dataloader is working here. This template creates
    # an instance of your dataloader and loads 20 instances from the dataset.
    # Fill in the missing part. This section is only run when the current file
    # is run and ignored when this file is imported.

    # This points to the root of the dataset
    data_root = '../data_set/data_set/data/'
    # This points to a file that contains the list of the filenames to be
    # loaded.
    test_list = '../data_set/data_set/test.str'
    print('[+] Init dataloader')
    # Fill in your dataset initializations
    testSet = SwappedDatasetLoader(test_list, data_root)
    print('[+] Create workers')
    loader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print('[*] Dataset size: ', len(loader))
    enu = enumerate(loader)
    for i in range(20):
        a = time.time()
        i, (images) = next(enu)
        b = time.time()
    #     # Uncomment to use a prettily printed version of a dict returned by the
    #     # dataloader.
        # print(images[0])
        printTensorList(images[0], True)
        print('[*] Time taken: ', b - a)
