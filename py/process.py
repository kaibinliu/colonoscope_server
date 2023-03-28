# -*- coding: utf-8 -*-
import os
import sys

import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

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

def process_image(image_path):
    model = UNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('E:/IDEA2018/IDEAworkplace3/colonoscope_server/model/unet_model-6.pth', map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    with torch.no_grad():
        image = Image.open(
            image_path).convert(
            'RGB')

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
        image = Image.fromarray(np.transpose((image.cpu().numpy() * 255).astype(np.uint8), (1, 2, 0)))
        # 将输出掩模张量数据转换为图像
        output_mask = Image.fromarray((output_mask.cpu().numpy() * 255).astype(np.uint8))
        # 获取物体的轮廓
        image_mask = output_mask.copy()
        output_mask = np.array(output_mask)
        contours, _ = cv2.findContours(output_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过小的碎屑点不视为病灶部分
        MIN_WIDTH = 10
        MIN_HEIGHT = 10

        # 针对每个轮廓获取其矩形边界框
        bounding_boxes = []
        for contour in contours:
            # 将轮廓转换为矩形边界框
            x, y, w, h = cv2.boundingRect(contour)
            # 判断矩形边界框的宽度和高度是否符合要求
            if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                bounding_boxes.append([x, y, w, h])
        # 在PIL图像上绘制矩形框
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        for bbox in bounding_boxes:
            draw.rectangle((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]), outline='red', width=3)

        # 返回PIL图像
        # return (image_mask, image)
        # 保存PIL图像
        path1 = 'E:/IDEA2018/IDEAworkplace3/colonoscope/public/images/maskImage/'
        path2 = 'E:/IDEA2018/IDEAworkplace3/colonoscope/public/images/bboxesImage/'
        # 获取文件名和扩展名
        filename = os.path.splitext(os.path.basename(image_path))[0]  # 获取文件名（不带扩展名）
        # extension = os.path.splitext(os.path.basename(image_path))[1]  # 获取扩展名（包括'.'）
        extension = '.jpeg'
        name = filename + extension
        image_mask.save(path1+name, format='JPEG')
        image.save(path2+name, format='JPEG')

if __name__ == '__main__':
    # 获取命令行参数
    image_path = sys.argv[1]
    # 调用函数
    result = process_image(image_path)
    # 输出结果
    # print(result)