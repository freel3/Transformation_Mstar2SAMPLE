'''
    Calculation of several quality evaluation indicators for generated images
'''
import os
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
import torch_fidelity

real_data_path = "result/real/"
fake_data_path1 = "result/our/"
fake_data_path2 = "result/cycle/"

def calculate_enl(image):
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    enl = (mean_intensity ** 2) / (std_intensity ** 2)
    return enl

def calculate_metrics(real_dir, fake_dir):
    real_images = sorted(os.listdir(real_dir))
    fake_images = sorted(os.listdir(fake_dir))

    ssim_values = []
    enl_values = []

    for real_image_name, fake_image_name in zip(real_images, fake_images):
        real_image_path = os.path.join(real_dir, real_image_name)
        fake_image_path = os.path.join(fake_dir, fake_image_name)

        real_image = io.imread(real_image_path, as_gray=True)
        fake_image = io.imread(fake_image_path, as_gray=True)

        # Calculate SSIM
        ssim_value = ssim(real_image, fake_image)
        ssim_values.append(ssim_value)

        # Calculate ENL for real and fake images and take the average
        enl_real = calculate_enl(real_image)
        enl_fake = calculate_enl(fake_image)
        average_enl = abs(enl_real - enl_fake)
        enl_values.append(average_enl)

    avg_ssim = np.mean(ssim_values)
    avg_enl = np.mean(enl_values)

    return avg_ssim, avg_enl

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    avg_ssim, avg_enl = calculate_metrics(real_data_path, fake_data_path1)
    metrics = torch_fidelity.calculate_metrics(
        input1=real_data_path,
        input2=fake_data_path1,
        cuda=True,
        fid=True,
        samples_find_deep=True
    )
    print("our Average SSIM:", avg_ssim)
    print("our Average DENL:", avg_enl)
    print("our FID Score:", metrics['frechet_inception_distance'])

    avg_ssim, avg_enl = calculate_metrics(real_data_path, fake_data_path2)
    metrics = torch_fidelity.calculate_metrics(
        input1=real_data_path,
        input2=fake_data_path2,
        cuda=True,
        fid=True,
        samples_find_deep=True
    )
    print("cycle Average SSIM:", avg_ssim)
    print("cycle Average DENL:", avg_enl)
    print("cycle FID Score:", metrics['frechet_inception_distance'])