#!/usr/bin/python3
'''
    Train model without ASC
'''
import argparse
import itertools
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import os
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
from scipy import io

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='database/',
                    help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', default='True', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--gpu_id', type=int, default=1, help='index of GPU to use')
opt = parser.parse_args()
print(opt)
# 设置可见的 CUDA 设备，仅限制为指定的 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

# 检查是否有可用的 CUDA 设备
device = torch.device("cuda" if torch.cuda.is_available() and opt.cuda else "cpu")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
# criterion_generation = torch.nn.L1Loss()
criterion_generation = torch.nn.MSELoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr/4, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr/4, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.ToTensor()]

dataset = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=False)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = batch['A'].to(device)  # 将batch中的'A'数据移动到指定的设备上（例如GPU）
        real_B = batch['B'].to(device)  # 将batch中的'B'数据移动到指定的设备上（例如GPU）

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        # 标签对比
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Generation loss
        loss_gen_A = criterion_generation(fake_A, real_A) * 10.0
        loss_gen_B = criterion_generation(fake_B, real_B) * 10.0

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + \
                 loss_gen_A +  loss_gen_B
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        if i % 1 == 0:
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_A = (loss_D_real + loss_D_fake)
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_real + loss_D_fake)
            loss_D_B.backward()

            optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G,
                    'loss_G_identity': (loss_identity_A + loss_identity_B),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                    'loss_D': (loss_D_A + loss_D_B),
                    'loss_Generation': (loss_gen_A + loss_gen_B)})

    print('loss_G:', loss_G.item())
    print('loss_G_identity:', (loss_identity_A + loss_identity_B).item())
    print('loss_G_GAN:', (loss_GAN_A2B + loss_GAN_B2A).item())
    print('loss_G_cycle:', (loss_cycle_ABA + loss_cycle_BAB).item())
    print('loss_D:', (loss_D_A + loss_D_B).item())
    print('loss_Generation:', (loss_gen_A + loss_gen_B).item())

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if (epoch>300 and epoch % 2 == 0):
        path1 = 'Transformation_Mstar2SAMPLE'
        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), path1 + '/output/model_cycle/netG_A2B_epoch{:d}.pth'.format(epoch + 1))
        torch.save(netG_B2A.state_dict(), path1 + '/output/model_cycle/netG_B2A_epoch{:d}.pth'.format(epoch + 1))
        torch.save(netD_A.state_dict(), path1 + '/output/model_cycle/netD_A_epoch{:d}.pth'.format(epoch + 1))
        torch.save(netD_B.state_dict(), path1 + '/output/model_cycle/netD_B_epoch{:d}.pth'.format(epoch + 1))

        # save images

        real_part = real_A[0, 0, :, :]
        imaginary_part = real_A[0, 1, :, :]
        abs_part = real_A[0, 2, :, :]
        # 合并为复数数据
        complex_data = torch.complex(real_part, imaginary_part)
        # print(complex_data)
        # 转换为 NumPy 数组
        complex_np = complex_data.detach().cpu().numpy().astype(np.complex128)
        abs_part_array = abs_part.detach().cpu().numpy()
        io.savemat(path1 + '/output/image_cycle/real_A_epoch{:d}.mat'.format(epoch + 1), {'complex_img': complex_np, 'abs_img': abs_part_array})

        real_part = real_B[0, 0, :, :]
        imaginary_part = real_B[0, 1, :, :]
        abs_part = real_B[0, 2, :, :]
        # 合并为复数数据
        complex_data = torch.complex(real_part, imaginary_part)
        # 转换为 NumPy 数组
        complex_np = complex_data.detach().cpu().numpy().astype(np.complex128)
        abs_part_array = abs_part.detach().cpu().numpy()
        io.savemat(path1 + '/output/image_cycle/real_B_epoch{:d}.mat'.format(epoch + 1), {'complex_img': complex_np, 'abs_img': abs_part_array})

        real_part = fake_A[0, 0, :, :]
        imaginary_part = fake_A[0, 1, :, :]
        abs_part = fake_A[0, 2, :, :]
        # 合并为复数数据
        complex_data = torch.complex(real_part, imaginary_part)
        # 转换为 NumPy 数组
        complex_np = complex_data.detach().cpu().numpy().astype(np.complex128)
        abs_part_array = abs_part.detach().cpu().numpy()
        io.savemat(path1 + '/output/image_cycle/fake_A_epoch{:d}.mat'.format(epoch + 1), {'complex_img': complex_np, 'abs_img': abs_part_array})

        real_part = fake_B[0, 0, :, :]
        imaginary_part = fake_B[0, 1, :, :]
        abs_part = fake_B[0, 2, :, :]
        # 合并为复数数据
        complex_data = torch.complex(real_part, imaginary_part)
        # 转换为 NumPy 数组
        complex_np = complex_data.detach().cpu().numpy().astype(np.complex128)
        abs_part_array = abs_part.detach().cpu().numpy()
        io.savemat(path1 + '/output/image_cycle/fake_B_epoch{:d}.mat'.format(epoch + 1), {'complex_img': complex_np, 'abs_img': abs_part_array})
        print("保存成功！！！")
###################################
