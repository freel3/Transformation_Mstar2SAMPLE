'''
    Batch visualization of mat format data
'''
import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import imageio
import os

def load_complex_image(file_path):
    data = loadmat(file_path)
    if 'complex_img' in data:
        complex_img = data['complex_img']
        abs_img = np.abs(complex_img)
        scaled_img = (abs_img - np.min(abs_img)) / (np.max(abs_img) - np.min(abs_img)) * 255
        return scaled_img.astype(np.uint8)
    else:
        raise ValueError("未找到复数图像数据 'complex_img'")

def load_complex_image_our(file_path):
    data = loadmat(file_path)
    if 'complex_img' in data:
        # complex_img = data['complex_img'] + data['ISAR']
        complex_img = data['ISAR']
        abs_img = np.abs(complex_img)
        scaled_img = (abs_img - np.min(abs_img)) / (np.max(abs_img) - np.min(abs_img)) * 255
        return scaled_img.astype(np.uint8)
    else:
        raise ValueError("未找到复数图像数据 'complex_img'")

def calculate_ssim(img1, img2):
    '''计算 SSIM'''
    if not img1.shape == img2.shape:
        raise ValueError('输入图像必须具有相同的尺寸。')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('错误的输入图像维度。')

# 计算ENL
def calculate_enl(image):
    mu = np.mean(image)
    sigma = np.std(image)
    enl = (mu ** 2) / (sigma ** 2)
    return enl

if __name__ == "__main__":
    # 定义文件夹路径和组数范围
    folder_path = "database/test_MSTAR/"
    # 获取文件夹下所有的 .mat 文件
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

    # 遍历所有 .mat 文件并处理
    for i, mat_file in enumerate(mat_files):
        mat_file_path = os.path.join(folder_path, mat_file)
        mat_file_path = f"test_data/realA/real_A_epoch{i}.mat"

        # 加载复数图像数据并转换为幅度图像
        img1 = load_complex_image(mat_file_path)

        # 保存幅度图像
        image1 = Image.fromarray(img1)
        img1_array = np.array(img1).astype(np.float32) / 255.0
        imageio.imwrite(f"result/mstar/mstar{i}.png", image1)
        imageio.imwrite(f"result/mstar_tif/mstar{i}.tiff", img1_array)


