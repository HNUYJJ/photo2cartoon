import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
import os

def get_inception_features(images, model, batch_size=32):
    """计算给定图像的 Inception 网络特征"""
    loader = DataLoader(images, batch_size=batch_size)
    features = []
    for img in loader:
        img = img.to(device)
        with torch.no_grad():
            pred = model(img)
        features.append(pred.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features


def calculate_fid(real_features, fake_features):
    """计算 FID 分数"""
    # 计算两组特征的均值和协方差
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    # 计算均值的差异的平方
    ssdiff = np.sum((mu1 - mu2) ** 2)

    # 计算协方差矩阵的平方根
    covmean = sqrtm(sigma1.dot(sigma2))

    # 检查并处理复数
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算 FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# 加载预训练的 InceptionV3 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
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

real_images_folder_A2B =  r'C:\Users\LITTLEWHITE\Desktop\photo2cartoon-master\dataset\photo2cartoon\testA'
generated_images_folder_A2B = r'C:\Users\LITTLEWHITE\Desktop\photo2cartoon-master\dataset\photo2cartoon\testA_result'

# 计算FID距离值

real_images = load_and_transform_images(real_images_folder_A2B)
fake_images = load_and_transform_images(generated_images_folder_A2B)

# 假设 real_images 和 fake_images 是两个包含图像张量的列表或 DataLoader
real_features = get_inception_features(real_images, inception_model)
fake_features = get_inception_features(fake_images, inception_model)

# 计算 FID
fid_score = calculate_fid(real_features, fake_features)
print(f"FID score: {fid_score}")