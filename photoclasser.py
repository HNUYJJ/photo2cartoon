import os
import shutil


def select_and_copy_every_nth_photo(src_dir, dest_dir, n=100):
    # 确保目标文件夹存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

        # 获取源文件夹中所有图片文件
    photos = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 根据文件名排序，以确保顺序正确
    photos.sort()

    # 初始化计数器
    count = 0
    for i in range(0, len(photos), n):
        # 选择每一百张中的第一张
        selected_photo = photos[i]

        # 构建目标路径
        dest_path = os.path.join(dest_dir, f'photo_{count:04d}.jpg')  # 使用四位数的编号格式

        # 复制文件
        shutil.copy2(selected_photo, dest_path)
        print(f"Copied {selected_photo} to {dest_path}")

        # 增加计数器
        count += 1

    # 使用示例


source_directory = r'C:\Users\LITTLEWHITE\Desktop\xinggan_face'  # 源数据集文件夹路径
destination_directory = r'C:\Users\LITTLEWHITE\Desktop\photo2cartoon-master\dataset\photo2cartoon\trainB'  # 目标文件夹路径
select_and_copy_every_nth_photo(source_directory, destination_directory)