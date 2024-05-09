import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from PIL import Image
import os
import torch
from torchvision.models.inception import Inception_V3_Weights

# 准备真实数据分布和生成模型的图像数据
def fid_calculate():
    real_images_folder_A2B = r'C:\Users\LITTLEWHITE\Desktop\photo2cartoon-master\dataset\photo2cartoon\testA'
    generated_images_folder_A2B = r'C:\Users\LITTLEWHITE\Desktop\photo2cartoon-master\dataset\photo2cartoon\testA_result'
    # real_images_folder_B2A = f'./dataset/{dataset}/testA'
    # generated_images_folder_B2A = f'./experiment/train-size256-ch32-True-lr0.0001-adv1-cyc50-id1-identity10-cam1000/{dataset}/test_B2A'
    path_A2B = [real_images_folder_A2B, generated_images_folder_A2B]
    # path_B2A = [real_images_folder_B2A, generated_images_folder_B2A]

    # 加载预训练的Inception-v3模型
    # 注意：这里使用weights参数而不是pretrained
    inception_model = torchvision.models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception_model.eval()

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(299),  # 注意：Inception-v3需要299x299的图像
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载并转换图像
    def load_and_transform_images(folder):
        images = []
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(folder, filename)).convert('RGB')
                img_t = transform(img)
                images.append(img_t)
        return torch.stack(images, dim=0)

        # 计算FID距离值
    fid_value_A2B = fid_score.calculate_fid_given_paths(path_A2B, batch_size=16, device='cpu', dims=2048)
    # fid_value_B2A = fid_score.calculate_fid_given_paths(path_B2A, batch_size=16, device='cpu', dims=2048)

    print('FID_A2B value:', fid_value_A2B)
    # print('FID_B2A value:', fid_value_B2A)

fid_calculate()