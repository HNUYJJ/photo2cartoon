import numpy as np
import torchvision
from scipy.linalg import sqrtm
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
from torchvision.models.inception import Inception_V3_Weights
import argparse

def KID_calculate(dataset):
    def get_inception_features(images, model, device):
        model.eval()
        with torch.no_grad():
            features = model(images.to(device)).detach().cpu().numpy()
        return features

    def calculate_kid(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2), disp=False)[0]
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        kid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return kid

    # 加载预训练的Inception模型
    device = torch.device("cpu")
    inception_model = torchvision.models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception_model.eval()
    inception_model.fc = torch.nn.Identity()
    inception_model.to(device)

    real_images_folder_A2B = r'C:\Users\LITTLEWHITE\Desktop\UGATIT类模型\UGATIT-pytorch-master\dataset\selfie2anime\trainB'
    generated_images_folder_A2B = r'C:\Users\LITTLEWHITE\Desktop\UGATIT类模型\UGATIT-pytorch-master\dataset\selfie2anime\testB'
    # real_images_folder_B2A = f'./dataset/{dataset}/testA'
    # generated_images_folder_B2A = f'./experiment/train-size256-ch32-True-lr0.0001-adv1-cyc50-id1-identity10-cam1000/{dataset}/test_B2A'

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

    real_images_A2B = load_and_transform_images(real_images_folder_A2B)
    gen_images_A2B = load_and_transform_images(generated_images_folder_A2B)

    # real_images_B2A = load_and_transform_images(real_images_folder_B2A)
    # gen_images_B2A = load_and_transform_images(generated_images_folder_B2A)

    # 假设 real_images 和 generated_images 是已经预处理好的图像张量
    # 提取特征
    real_features_A2B = get_inception_features(real_images_A2B, inception_model, device)
    generated_features_A2B = get_inception_features(gen_images_A2B, inception_model, device)

    # real_features_B2A = get_inception_features(real_images_B2A, inception_model, device)
    # generated_features_B2A = get_inception_features(gen_images_B2A, inception_model, device)

    # 计算均值和协方差
    mu_real_A2B = np.mean(real_features_A2B, axis=0)
    sigma_real_A2B = np.cov(real_features_A2B, rowvar=False)
    mu_gen_A2B = np.mean(generated_features_A2B, axis=0)
    sigma_gen_A2B = np.cov(generated_features_A2B, rowvar=False)

    # mu_real_B2A = np.mean(real_features_B2A, axis=0)
    # sigma_real_B2A = np.cov(real_features_B2A, rowvar=False)
    # mu_gen_B2A = np.mean(generated_features_B2A, axis=0)
    # sigma_gen_B2A = np.cov(generated_features_B2A, rowvar=False)

    # 计算KID
    kid_value_A2B = calculate_kid(mu_real_A2B, sigma_real_A2B, mu_gen_A2B, sigma_gen_A2B)
    # kid_value_B2A = calculate_kid(mu_real_B2A, sigma_real_B2A, mu_gen_B2A, sigma_gen_B2A)

    print(f"{dataset}_KID_A2B:", kid_value_A2B)
    # print(f"{dataset}_KID_B2A:", kid_value_B2A)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Calculate KID score for a given dataset.')
    # parser.add_argument('--input_path', type=str, required=True, help='Path to the input dataset.')
    # parser.add_argument('--output_path', type=str, required=True, help='Path to the generated/output images.')
    # parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset.')
    # args = parser.parse_args()

    KID_calculate("slefie2anime")
