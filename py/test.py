# -*- coding: utf-8 -*-
# import os
# # 获取当前脚本的绝对路径
# script_dir = os.path.dirname(os.path.abspath(__file__))
#
# # 引用相对路径下的文件
# file_path = os.path.join(script_dir, "../model/unet_model-6.pth")
#
# print(file_path)
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dropout = torch.nn.Dropout(0.5)

        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3_3 = torch.nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3_4 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2_3 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv2_4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1_3 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv1_4 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv_out = torch.nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        conv1 = torch.nn.functional.relu(self.conv1_1(x))
        conv1 = torch.nn.functional.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = torch.nn.functional.relu(self.conv2_1(pool1))
        conv2 = torch.nn.functional.relu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)

        conv3 = torch.nn.functional.relu(self.conv3_1(pool2))
        conv3 = torch.nn.functional.relu(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)

        conv4 = torch.nn.functional.relu(self.conv4_1(pool3))
        conv4 = torch.nn.functional.relu(self.conv4_2(conv4))
        drop4 = self.dropout(conv4)

        upconv3 = self.upconv3(drop4)
        concat3 = torch.cat([upconv3, conv3], dim=1)
        conv3 = torch.nn.functional.relu(self.conv3_3(concat3))
        conv3 = torch.nn.functional.relu(self.conv3_4(conv3))

        upconv2 = self.upconv2(conv3)
        concat2 = torch.cat([upconv2, conv2], dim=1)
        conv2 = torch.nn.functional.relu(self.conv2_3(concat2))
        conv2 = torch.nn.functional.relu(self.conv2_4(conv2))

        upconv1 = self.upconv1(conv2)
        concat1 = torch.cat([upconv1, conv1], dim=1)
        conv1 = torch.nn.functional.relu(self.conv1_3(concat1))
        conv1 = torch.nn.functional.relu(self.conv1_4(conv1))

        out = self.conv_out(conv1)

        return out

def process_image(image, model):
    # resize image and mask to the same size
    new_size = (256, 256)
    image = image.resize(new_size)
    image = transforms.ToTensor()(image)
    image = image.to(device)
    input_image = image.unsqueeze(0)
    output_mask = model(input_image)

    # 模型输出为 output_mask，shape 为 [1, 256, 256]
    # 先将 output_mask 转换为 [256, 256] 的 numpy 数组
    output_mask = np.squeeze(output_mask)
    # 对输出进行 sigmoid 处理
    output_sigmoid = torch.sigmoid(output_mask)
    # 将概率值与阈值进行比较，得到二进制掩模张量
    output_mask = (output_sigmoid > 0.5).float()
    # 将张量数据转换为图像(PIL)
    # image = Image.fromarray(np.transpose((image.cpu().numpy() * 255).astype(np.uint8), (1, 2, 0)))
    # 将输出掩模张量数据转换为图像
    output_mask = Image.fromarray((output_mask.cpu().numpy() * 255).astype(np.uint8))
    return output_mask

def calculate_accuracy(generated_mask, annotated_mask):
    # 计算准确率
    correct_pixels = np.sum(np.logical_and(generated_mask, annotated_mask))
    total_pixels = np.sum(annotated_mask)
    accuracy = (correct_pixels / total_pixels) * 100
    return accuracy

def calculate_dice_coefficient(generated_mask, annotated_mask):
    # 计算Dice系数
    intersection = np.sum(np.logical_and(generated_mask, annotated_mask))
    total_pixels = np.sum(generated_mask) + np.sum(annotated_mask)
    dice_coefficient = (2 * intersection) / total_pixels
    return dice_coefficient

if __name__ == '__main__':
    model = UNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("../model/unet_model-6.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    # 定义文件夹路径
    folder_path = "D:/学习资料/学校/毕业相关/毕业设计/训练集/kvasir训练集/sessile-main-Kvasir-SEG"
    image_folder = folder_path + '/images'
    mask_folder = folder_path + '/masks'
    # print(folder_path,image_folder,mask_folder)
    # 初始化结果列表
    accuracies = []
    dice_coefficients = []
    with torch.no_grad():
        # 遍历文件夹中的图像
        i = 0
        for filename in os.listdir(image_folder):
            # 构建图像和掩模文件的路径
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)

            # 打开图像文件并调整大小为256x256
            image = Image.open(image_path).convert('RGB')
            image = image.resize((256, 256))

            # 处理图像得到二进制掩模图像
            generated_mask = process_image(image, model)
            generated_mask = generated_mask.resize((256,256))
            generated_mask = np.array(generated_mask)
            generated_mask = np.uint8(generated_mask > 0)

            # 打开掩模图像文件并调整大小为256x256
            annotated_mask = Image.open(mask_path).convert('L')
            annotated_mask = annotated_mask.resize((256, 256))

            # 将掩模图像转换为二进制掩模图像
            annotated_mask = np.array(annotated_mask)
            annotated_mask = np.uint8(annotated_mask > 0)

            # 计算准确率和Dice系数
            accuracy = calculate_accuracy(generated_mask, annotated_mask)
            dice_coefficient = calculate_dice_coefficient(generated_mask, annotated_mask)

            # 将结果添加到列表中
            accuracies.append(accuracy)
            dice_coefficients.append(dice_coefficient)

            # 打印单对图片计算结果
            i += 1
            print(f"图序号：{i}" + "    " + f"本图准确率: {accuracy}%" + "    " + f"本图Dice系数: {dice_coefficient}")
        # 计算平均准确率和平均Dice系数
        mean_accuracy = np.mean(accuracies)
        mean_dice_coefficient = np.mean(dice_coefficients)
        # 打印结果
        print(f"平均准确率: {mean_accuracy}%")
        print(f"平均Dice系数: {mean_dice_coefficient}")
        # 绘制准确率折线图
        # 设置全局字体
        plt.rcParams['font.family'] = 'SimHei'
        plt.plot(accuracies)
        plt.xlabel('样本编号')
        plt.ylabel('准确率')
        plt.title('准确率随样本编号的变化')
        plt.show()

        # 绘制Dice系数折线图
        plt.plot(dice_coefficients)
        plt.xlabel('样本编号')
        plt.ylabel('Dice系数')
        plt.title('Dice系数随样本编号的变化')
        plt.show()

