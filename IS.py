import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import entropy
from PIL import Image
import os

def get_inception_probs(images, model, batch_size=32):
    """计算给定图像的 Inception 网络输出概率"""
    loader = DataLoader(images, batch_size=batch_size)
    probs = []
    for img in loader:
        img = img.to(device)
        with torch.no_grad():
            pred = model(img)
            # 使用 softmax 获取概率分布
            p = torch.nn.functional.softmax(pred, dim=1)
        probs.append(p.cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    return probs

def calculate_inception_score(probs, splits=10):
    """计算 Inception Score"""
    # 分割数据以计算分数
    scores = []
    chunk_size = len(probs) // splits
    for i in range(splits):
        part = probs[i * chunk_size: (i + 1) * chunk_size, :]
        # 计算 p(y|x) 的熵
        pyx = np.mean(part, axis=0)
        # 计算 KL 散度
        scores.append(entropy(pyx, np.mean(probs, axis=0)))
    # 计算最终的 Inception Score
    is_score = np.exp(np.mean(scores))
    return is_score

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

generated_images_folder_A2B = r'C:\Users\LITTLEWHITE\Desktop\photo2cartoon-master\dataset\photo2cartoon\testA_result'
fake_images = load_and_transform_images(generated_images_folder_A2B)

# 假设 fake_images 是一个包含生成图像张量的列表或 DataLoader
probs = get_inception_probs(fake_images, inception_model)

# 计算 Inception Score
is_score = calculate_inception_score(probs)
print(f"Inception Score: {is_score}")