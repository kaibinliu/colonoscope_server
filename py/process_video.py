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
    return image_mask,image

if __name__ == '__main__':
    # 获取命令行参数
    video_path = sys.argv[1]
    model = UNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('E:/IDEA2018/IDEAworkplace3/colonoscope_server/model/unet_model-6.pth', map_location=device)

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    with torch.no_grad():
        cap = cv2.VideoCapture(video_path)

        # 获取视频的FPS，宽度和高度
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = 256
        height = 256

        # 定义输出视频文件的路径
        filename = os.path.splitext(os.path.basename(video_path))[0]  # 获取文件名（不带扩展名）
        mask_video_path = 'E:/IDEA2018/IDEAworkplace3/colonoscope/public/video/maskVideo/'+filename+'.mp4'
        bbox_video_path = 'E:/IDEA2018/IDEAworkplace3/colonoscope/public/video/bboxesVideo/'+filename+'.mp4'

        # 创建输出视频对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mask_video_writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height))
        bbox_video_writer = cv2.VideoWriter(bbox_video_path, fourcc, fps, (width, height))

        # 处理的第一帧
        ret, frame = cap.read()
        if not ret:
            raise Exception("Can't read the first frame")

        # 将OpenCV帧转换为PIL图像
        prev_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 处理第一帧并获取输出
        prev_mask_image, prev_bbox_image = process_image(prev_image, model)

        # 将PIL图像转换回OpenCV帧
        prev_mask_frame = cv2.cvtColor(np.array(prev_mask_image), cv2.COLOR_GRAY2BGR)
        prev_bbox_frame = cv2.cvtColor(np.array(prev_bbox_image), cv2.COLOR_RGB2BGR)

        # 写入输出视频
        mask_video_writer.write(prev_mask_frame)
        bbox_video_writer.write(prev_bbox_frame)

        # 循环处理视频中的每个帧
        threshold = 10
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 将OpenCV帧转换为PIL图像
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 处理图像并获取输出
            mask_image, bbox_image = None, None
            if np.sum(np.abs(np.array(image) - np.array(prev_image))) >= threshold:
                mask_image, bbox_image = process_image(image, model)
                prev_image = image

            if mask_image is not None and bbox_image is not None:
                # 将PIL图像转换回OpenCV帧
                mask_frame = cv2.cvtColor(np.array(mask_image), cv2.COLOR_GRAY2BGR)
                bbox_frame = cv2.cvtColor(np.array(bbox_image), cv2.COLOR_RGB2BGR)

                # 写入输出视频
                mask_video_writer.write(mask_frame)
                bbox_video_writer.write(bbox_frame)

        # 释放资源
        cap.release()
        mask_video_writer.release()
        bbox_video_writer.release()
        cv2.destroyAllWindows()
