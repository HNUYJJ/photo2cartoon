import os
import cv2
import torch
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess
import tkinter as tk
from tkinter import filedialog

# parser = argparse.ArgumentParser()
# parser.add_argument('--photo_path', type=str, help='input photo path')
# parser.add_argument('--save_path', type=str, help='cartoon save path')
# args = parser.parse_args()

# os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)

        assert os.path.exists('./models/photo2cartoon_weights.pt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        params = torch.load('./models/photo2cartoon_weights.pt', map_location=self.device)
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')

    def inference(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None

        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon

# if __name__ == '__main__':
#     img = cv2.cvtColor(cv2.imread(args.photo_path), cv2.COLOR_BGR2RGB)
#     c2p = Photo2Cartoon()
#     cartoon = c2p.inference(img)
#     if cartoon is not None:
#         cv2.imwrite(args.save_path, cartoon)
#         print('Cartoon portrait has been saved successfully!')

# import os
# import cv2
# import torch
# import numpy as np
# from models import ResnetGenerator
# import argparse
# from utils import Preprocess
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--input_dir', type=str, help='input directory path containing photos')
# parser.add_argument('--output_dir', type=str, help='output directory path to save cartoons')
# args = parser.parse_args()
#
# os.makedirs(args.output_dir, exist_ok=True)
#
# class Photo2Cartoon:
#     def __init__(self):
#         self.pre = Preprocess()
#         self.device = torch.device("cpu")
#         self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
#
#         assert os.path.exists('./models/photo2cartoon_weights.pt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
#         params = torch.load('./models/photo2cartoon_weights.pt', map_location=self.device)
#         self.net.load_state_dict(params['genA2B'])
#         print('[Step1: load weights] success!')
#
#     def inference(self, img):
#         # face alignment and segmentation
#         face_rgba = self.pre.process(img)
#         if face_rgba is None:
#             print('[Step2: face detect] can not detect face!!!')
#             return None
#
#         print('[Step2: face detect] success!')
#         face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
#         face = face_rgba[:, :, :3].copy()
#         mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
#         face = (face*mask + (1-mask)*255) / 127.5 - 1
#
#         face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
#         face = torch.from_numpy(face).to(self.device)
#
#         # inference
#         with torch.no_grad():
#             cartoon = self.net(face)[0][0]
#
#         # post-process
#         cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
#         cartoon = (cartoon + 1) * 127.5
#         cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
#         cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
#         print('[Step3: photo to cartoon] success!')
#         return cartoon
#
# if __name__ == '__main__':
#     c2p = Photo2Cartoon()
#
#     # 遍历输入目录中的所有文件
#     for filename in os.listdir(args.input_dir):
#         if filename.endswith('.jpg') or filename.endswith('.png'):  # 根据需要添加其他图像格式
#             input_path = os.path.join(args.input_dir, filename)
#             output_path = os.path.join(args.output_dir, filename)
#
#             img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
#             cartoon = c2p.inference(img)
#             if cartoon is not None:
#                 cv2.imwrite(output_path, cartoon)
#                 print(f'Cartoon portrait for {filename} has been saved successfully!')

    # # 创建一个Tkinter窗口实例（隐藏）以支持文件对话框
    # root = tk.Tk()
    # root.withdraw()  # 隐藏主窗口
    #
    # # 弹出文件选择对话框并获取用户选择的文件路径
    # args.photo_path = filedialog.askopenfilename(title="请选择一张照片",
    #                                              filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp"),
    #                                                         ("所有文件", "*.*")])
    #
    # # 如果用户选择了文件（没有点击取消）
    # if args.photo_path:
    #     # 确保有保存路径，如果没有提供，则默认为与输入图片相同的目录，文件名添加"_cartoon.jpg"
    #     if not args.save_path:
    #         directory, filename = os.path.split(args.photo_path)
    #         base_filename, file_extension = os.path.splitext(filename)
    #         args.save_path = os.path.join(directory, f"{base_filename}_cartoon.jpg")
    #
    #         # 读取并转换图片格式
    #     img = cv2.imread(args.photo_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     # 执行卡通化处理
    #     c2p = Photo2Cartoon()
    #     cartoon = c2p.inference(img)
    #
    #     # 如果卡通化成功，保存图片
    #     if cartoon is not None:
    #         cv2.imwrite(args.save_path, cartoon)
    #         print('Cartoon portrait has been saved successfully!')
    #
    #         # 销毁Tkinter窗口实例
    # root.destroy()


