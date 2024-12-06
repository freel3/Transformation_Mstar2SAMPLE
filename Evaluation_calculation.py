'''
    Batch computing SSIM and ENL for images
'''
import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import imageio

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
    folder_path = "/home/sda/fengrui/python_project/PyTorch-CycleGAN-master/test_data"
    num_images = 282  # 要处理的图像组数

    max_ssim = -np.inf
    best_group_id = -1

    # 对第1到500组图像进行 SSIM 计算
    for i in range(1, num_images):
        img1_path = f"{folder_path}/realB/real_B_epoch{i}.mat"
        img2_path = f"{folder_path}/fakeB_our/fake_B_epoch{i}.mat"
        img3_path = f"{folder_path}/fakeB_cycle/fake_B_epoch{i}.mat"

        # 加载复数图像数据并缩放到 [0, 255]
        img1 = load_complex_image(img1_path)
        img2 = load_complex_image(img2_path)
        img3 = load_complex_image(img3_path)

        # 计算 SSIM
        ssim_value = calculate_ssim(img1, img2)

        # 打印 SSIM 值
        if ssim_value > 0.8:
            print(f"{i}  SSIM 值: {ssim_value:.4f}")
            ENL = calculate_enl(img2)
            print(f"     ENL  值: {ENL:.4f}")
            # 保存幅度图像
            image1 = Image.fromarray(img1)
            img1_array = np.array(img1).astype(np.float32) / 255.0
            imageio.imwrite(f"/home/sda/fengrui/python_project/PyTorch-CycleGAN-master/result/real/real{i}.png", image1)
            imageio.imwrite(f"/home/sda/fengrui/python_project/PyTorch-CycleGAN-master/result/real_tif/real{i}.tiff", img1_array)
            image2 = Image.fromarray(img2)
            img2_array = np.array(img2).astype(np.float32) / 255.0
            imageio.imwrite(f"/home/sda/fengrui/python_project/PyTorch-CycleGAN-master/result/our/fake{i}.png", image2)
            imageio.imwrite(f"/home/sda/fengrui/python_project/PyTorch-CycleGAN-master/result/our_tif/fake{i}.tiff", img2_array)
            image3 = Image.fromarray(img3)
            img3_array = np.array(img3).astype(np.float32) / 255.0
            imageio.imwrite(f"/home/sda/fengrui/python_project/PyTorch-CycleGAN-master/result/cycle/fake{i}.png", image3)
            imageio.imwrite(f"/home/sda/fengrui/python_project/PyTorch-CycleGAN-master/result/cycle_tif/fake{i}.tiff", img3_array)

        # 更新最大的 SSIM 值和对应的图像组编号
        if ssim_value > max_ssim:
            max_ssim = ssim_value
            best_group_id = i

    # 打印最大的 SSIM 值和对应的图像组编号
    print(f"最大 SSIM 值: {max_ssim:.4f}")
    print(f"对应图像组编号: {best_group_id}")
