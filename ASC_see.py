import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio

'''
    View ASC extraction results
'''
# 设置路径
input_dir = 'data/ASC_zhu/'  # 输入数据目录
output_dir = 'data/ASC_zhu/'  # 输出保存路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历文件夹中的.mat文件
for filename in os.listdir(input_dir):
    if filename.endswith('.mat'):
        mat_path = os.path.join(input_dir, filename)

        # 加载.mat文件
        mat_contents = scipy.io.loadmat(mat_path)

        image = np.zeros((128, 128))
        ASC = mat_contents['ASC']
        m, n = ASC.shape
        real = np.real(ASC)
        ASC_abs = np.abs(ASC)
        for i in range(0, n):
            image[int((real[1, i] + 19.2) / 0.3), int((real[0, i] + 19.2) / 0.3)] = ASC_abs[3, i]

        # 转换为 uint8 类型，以便保存为图像
        image_to_save = (image * 20).clip(0, 255).astype(np.float32)

        # 保存图像，文件名保持不变
        output_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.tiff')

        # 使用Pillow保存图像
        imageio.imwrite(output_path, image_to_save)

        print(f'Saved image: {output_path}')
# 设置路径
input_dir = 'data/complex_zhu/'  # 输入数据目录
output_dir = 'data/complex_zhu/'  # 输出保存路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历文件夹中的.mat文件
for filename in os.listdir(input_dir):
    if filename.endswith('.mat'):
        mat_path = os.path.join(input_dir, filename)

        # 加载.mat文件
        mat_contents = scipy.io.loadmat(mat_path)

        # 获取复数数据
        sar_data_complex = mat_contents['complex_img']

        abs_image = abs(sar_data_complex)

        # 转换为 uint8 类型，以便保存为图像
        image_to_save = (abs_image/5).clip(0, 255).astype(np.float32)

        # 保存图像，文件名保持不变
        output_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.tiff')

        # 使用Pillow保存图像
        imageio.imwrite(output_path, image_to_save)

        print(f'Saved image: {output_path}')

