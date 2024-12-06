'''
    Loading training and testing data
'''
import glob
import random
import os
import scipy
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        if mode == 'train':
            self.files_A = sorted(glob.glob(os.path.join(root, mode + '_MSTAR') + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, mode + '_SAMPLE') + '/*.*'))
            self.ASCfiles_A = sorted(glob.glob(os.path.join(root, mode + '_ASC_MSTAR') + '/*.*'))
            self.ASCfiles_B = sorted(glob.glob(os.path.join(root, mode + '_ASC_SAMPLE') + '/*.*'))
        if mode == 'test':
            self.files_A = sorted(glob.glob(os.path.join(root) + '/*.*'))
            # self.files_B = sorted(glob.glob(os.path.join(root, mode + '_SAMPLE') + '/*.*'))
            # self.ASCfiles_A = sorted(glob.glob(os.path.join(root + 't72_ASC') + '/*.*'))

    def __getitem__(self, index):
        index_A = index % len(self.files_A)
        self.mat_A = scipy.io.loadmat(self.files_A[index_A])
        self.complex_A = self.mat_A['complex_img']
        amplitude = np.abs(self.complex_A)
        min_val = np.min(amplitude)
        max_val = np.max(amplitude)
        scaled_amplitude = ((amplitude - min_val) / (max_val - min_val))
        scaled_complex_data = scaled_amplitude * np.exp(1j * np.angle(self.complex_A))
        real_part_A = np.real(scaled_complex_data)
        imag_part_A = np.imag(scaled_complex_data)
        real_part_A = self.transform(Image.fromarray(real_part_A))
        imag_part_A = self.transform(Image.fromarray(imag_part_A))
        amplitude_A = self.transform(Image.fromarray(scaled_amplitude))
        data_A = torch.cat([real_part_A, imag_part_A, amplitude_A],dim=0)


        self.ASC_A = scipy.io.loadmat(self.ASCfiles_A[index_A])
        self.complexASC_A = self.ASC_A['ASC']
        self.complexISAR_A = self.ASC_A['ISAR']
        m, n = self.complexASC_A.shape
        real = np.real(self.complexASC_A)
        imag = np.imag(self.complexASC_A)
        real_ISAR = np.real(self.complexISAR_A)
        imag_ISAR = np.imag(self.complexISAR_A)
        image = np.zeros((7, real_ISAR.shape[0], real_ISAR.shape[0]))
        for i in range(0, n):
            image[0, int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = real[2, i]    # L
            image[1, int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = real[3, i]    # PhiH
            image[2, int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = real[4, i]    # Alpha
            image[3, int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = real[5, i]    # A
            image[4, int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = imag[5, i]  # A
        image[5,...] = real_ISAR
        image[6,...] = imag_ISAR
        A0 = self.transform(Image.fromarray(image[0,...]))
        A1 = self.transform(Image.fromarray(image[1, ...]))
        A2 = self.transform(Image.fromarray(image[2, ...]))
        A3 = self.transform(Image.fromarray(image[3, ...]))
        A4 = self.transform(Image.fromarray(image[4, ...]))
        A5 = self.transform(Image.fromarray(image[5, ...]))
        A6 = self.transform(Image.fromarray(image[6, ...]))


        ASC_A = torch.cat([real_part_A, imag_part_A, amplitude_A, A5, A6, A0, A1, A2, A3, A4], dim=0)

        if self.unaligned:
            index_B = random.randint(0, len(self.files_B) - 1)
            self.mat_B = scipy.io.loadmat(self.files_B[index_B])
            self.complex_B = self.mat_B['complex_img']
            amplitude = np.abs( self.complex_B)
            # normalized_amplitude = amplitude / np.max(amplitude)
            # scaled_amplitude = normalized_amplitude * 255
            # scaled_complex_data = scaled_amplitude * np.exp(1j * np.angle(self.complex_B))
            real_part_B = np.real(self.complex_B)
            imag_part_B = np.imag(self.complex_B)
            real_part_B = self.transform(Image.fromarray(real_part_B))
            imag_part_B = self.transform(Image.fromarray(imag_part_B))
            amplitude_B = self.transform(Image.fromarray(amplitude))

            data_B = torch.cat([real_part_B, imag_part_B, amplitude_B],dim=0)
        else:
            index_B = index % len(self.files_B)
            self.mat_B = scipy.io.loadmat(self.files_B[index_B])
            self.complex_B = self.mat_B['complex_img']
            amplitude = np.abs(self.complex_B)
            min_val = np.min(amplitude)
            max_val = np.max(amplitude)
            scaled_amplitude = ((amplitude - min_val) / (max_val - min_val))
            scaled_complex_data = scaled_amplitude * np.exp(1j * np.angle(self.complex_B))
            real_part_B = np.real(scaled_complex_data)
            imag_part_B = np.imag(scaled_complex_data)
            real_part_B = self.transform(Image.fromarray(real_part_B))
            imag_part_B = self.transform(Image.fromarray(imag_part_B))
            amplitude_B = self.transform(Image.fromarray(scaled_amplitude))

            data_B = torch.cat([real_part_B, imag_part_B, amplitude_B], dim=0)


            image = np.zeros((7, 128, 128))
            self.ASC_B = scipy.io.loadmat(self.ASCfiles_B[index_B])
            self.complexASC_B = self.ASC_B['ASC']
            self.complexISAR_B = self.ASC_B['ISAR']
            m, n = self.complexASC_B.shape
            real = np.real(self.complexASC_B)
            imag = np.imag(self.complexASC_B)
            real_ISAR = np.real(self.complexISAR_B)
            imag_ISAR = np.imag(self.complexISAR_B)
            for i in range(0, n):
                image[0, int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = real[2, i]
                image[1, int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = real[3, i]
                image[2, int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = real[4, i]
                image[3, int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = real[5, i]  # A
                image[4, int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = imag[5, i]  # A
            image[5, ...] = real_ISAR
            image[6, ...] = imag_ISAR
            B0 = self.transform(Image.fromarray(image[0, ...]))
            B1 = self.transform(Image.fromarray(image[1, ...]))
            B2 = self.transform(Image.fromarray(image[2, ...]))
            B3 = self.transform(Image.fromarray(image[3, ...]))
            B4 = self.transform(Image.fromarray(image[4, ...]))
            B5 = self.transform(Image.fromarray(image[5, ...]))
            B6 = self.transform(Image.fromarray(image[6, ...]))

            ASC_B = torch.cat([real_part_B, imag_part_B, amplitude_B, B5, B6, B0, B1, B2, B3, B4], dim=0)

        return {'A': data_A, 'B': data_B, 'ASC_A': ASC_A, 'ASC_B': ASC_B}
        # return {'ASC_A': ASC_A}
        # return {'A': data_A}

    def __len__(self):
        return len(self.files_A)