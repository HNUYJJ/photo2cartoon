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


def calculate_kid(real_features, fake_features, num_subsets=100, subset_size=100):
    """计算 KID 分数"""
    real_features = np.array(real_features)
    fake_features = np.array(fake_features)
    kid_scores = []

    for _ in range(num_subsets):
        real_subset = real_features[np.random.choice(len(real_features), subset_size, replace=False)]
        fake_subset = fake_features[np.random.choice(len(fake_features), subset_size, replace=False)]

        real_cov = np.cov(real_subset, rowvar=False)
        fake_cov = np.cov(fake_subset, rowvar=False)
        mean_diff = np.mean(real_subset, axis=0) - np.mean(fake_subset, axis=0)

        cov_mean = sqrtm(real_cov @ fake_cov)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        kid_score = mean_diff @ mean_diff + np.trace(real_cov + fake_cov - 2 * cov_mean)
        kid_scores.append(kid_score)

    return np.mean(kid_scores)

real_images_folder_A2B =  r'C:\Users\LITTLEWHITE\Desktop\UGATIT类模型\UGATIT-pytorch-master\dataset\selfie2anime\trainB'
generated_images_folder_A2B = r'C:\Users\LITTLEWHITE\Desktop\UGATIT类模型\UGATIT-pytorch-master\dataset\selfie2anime\testB'

# 加载预训练的 InceptionV3 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model = models.inception_v3(pretrained=True).to(device)
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


# 计算KID距离值

real_images = load_and_transform_images(real_images_folder_A2B)
fake_images = load_and_transform_images(generated_images_folder_A2B)

# 假设 real_images 和 fake_images 是两个包含图像张量的列表或 DataLoader
real_features = get_inception_features(real_images, inception_model)
fake_features = get_inception_features(fake_images, inception_model)

# 计算 KID
kid_score = calculate_kid(real_features, fake_features)
print(f"KID score: {kid_score}")

