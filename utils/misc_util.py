import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
# pip install tensorboard
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())
        return output


def main():
    # Parameters and DataLoaders
    input_size = 5
    output_size = 2
    batch_size = 3000000
    data_size = 1000000000

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available() :#and GPU_ID > -1:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            MULTI_GPU = True
        else:
            device = torch.device('cuda:0')   #{}'.format(args.gpu_id))
            print('device : ',device)
            MULTI_GPU = False
    else:
        device = torch.device('cpu')
        print('device : ',device)

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
    
    model = Model(input_size, output_size)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    print('before loop ')
    step = 1
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        print("Outside: input size", input.size(), "output_size", output.size())

        angle_rad = step * math.pi / 180
        writer.add_scalars('loss and accuracy', 
        {'loss': math.sin(angle_rad), 'accuracy': math.cos(angle_rad)}, step)
        writer.flush()
        step+= 1

    writer.close()

    # for n_iter in range(100):
    #     writer.add_scalar('Loss/train_check', np.random.random(), n_iter)
    #     # writer.add_scalar('Loss/test', np.random.random(), n_iter)
    #     # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    #     # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

    # for step in range(-360, 360):
    #     angle_rad = step * math.pi / 180
    #     writer.add_scalar('sin', math.sin(angle_rad), step)
    #     writer.add_scalar('cos', math.cos(angle_rad), step)
    #     writer.add_scalars('sin and cos', {'sin': math.sin(angle_rad), 'cos': math.cos(angle_rad)}, step)
    # writer.close()



if __name__ == '__main__':
    main()

