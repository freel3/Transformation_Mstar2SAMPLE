'''
    Transform test data using cyclegan model incorporating ASC features
'''
import argparse
import sys
import os
from scipy import io
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='database', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=10, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=10, help='number of channels of output data')
parser.add_argument('--size', type=int, default=128, help='size of the data (squared assumed)')
parser.add_argument('--cuda', default='True', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default="output/model_our/netG_A2B_epoch477.pth", help='A2B generator checkpoint file')
parser.add_argument('--gpu_id', type=int, default=1, help='index of GPU to use')
opt = parser.parse_args()
print(opt)

# 设置可见的 CUDA 设备，仅限制为指定的 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

###### Definition of variables ######
# Network
netG_A2B = Generator(opt.input_nc, opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()

# Load state dict
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))

# Set model's test mode
netG_A2B.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, 3, opt.size, opt.size)
# Dataset loader
transforms_ = [transforms.Resize(128),
                transforms.ToTensor()]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test', unaligned=False),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
print(len(dataloader))
###################################

###### Testing######

# Create output dir if they don't exist
if not os.path.exists('test_data/fakeB'):
    os.makedirs('test_data/fakeB')

if not os.path.exists('test_data/realB'):
    os.makedirs('test_data/realB')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['ASC_A']))
    # real_A = Variable(input_A.copy_(batch['A']))    # no ASC
    # real_B = Variable(input_B.copy_(batch['B']))
    # Generate output
    # MSTAR to SAMPLE
    fake_B = netG_A2B(real_A).data

    # Save image files
    real_part = fake_B[0, 0, :, :]
    imaginary_part = fake_B[0, 1, :, :]
    abs_part = fake_B[0, 2, :, :]
    ISAR_real = fake_B[0, 3, :, :]
    ISAR_imag = fake_B[0, 4, :, :]
    # 合并为复数数据
    complex_data = torch.complex(real_part, imaginary_part)
    complex_ISAR = torch.complex(ISAR_real, ISAR_imag)
    # 转换为 NumPy 数组
    complex_np = complex_data.cpu().numpy().astype(np.complex128)
    abs_part_array = abs_part.detach().cpu().numpy()
    ISAR_array = complex_ISAR.cpu().numpy().astype(np.complex128)
    io.savemat('output/mstar_classifier/add_fake/t72/fake_epoch{:d}.mat'.format(i+1), {'complex_img': complex_np, 'abs_img': abs_part_array, 'ISAR': ISAR_array})

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
