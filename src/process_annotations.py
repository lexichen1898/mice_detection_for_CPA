import os
import json
import cv2
import numpy as np
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import random

class AnnotationProcessor:
    def __init__(self, video_dir, annotation_dir, output_dir):
        self.video_dir = Path(video_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.frames_dir = self.output_dir / 'frames'
        self.labels_dir = self.output_dir / 'labels'
        self.vis_dir = self.output_dir / 'vis'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames(self, video_path):
        """从视频中提取帧"""
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = self.frames_dir / f"{video_path.stem}_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1
            
        cap.release()
        return frame_count
    
    def convert_cvat_to_yolo(self, annotation_path, frame_count, img_width, img_height):
        """将CVAT XML标注转换为YOLO格式"""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # 创建帧到标注的映射
        frame_annotations = {}
        for frame_idx in range(frame_count):
            frame_annotations[frame_idx] = []
        
        # 处理每个标注
        for track in root.findall('track'):
            for box in track.findall('box'):
                frame_idx = int(box.get('frame'))
                if frame_idx >= frame_count:
                    continue
                
                # 获取边界框坐标
                x1 = float(box.get('xtl'))
                y1 = float(box.get('ytl'))
                x2 = float(box.get('xbr'))
                y2 = float(box.get('ybr'))
                
                # 转换为YOLO格式 (x_center, y_center, width, height)
                x_center = (x1 + x2) / (2 * img_width)
                y_center = (y1 + y2) / (2 * img_height)
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                # 确保值在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # 添加到对应帧的标注列表
                frame_annotations[frame_idx].append([0, x_center, y_center, width, height])
        
        return frame_annotations
    
    def visualize_annotation(self, frame_path, label_path, vis_path):
        """可视化标注"""
        # 读取图像
        img = cv2.imread(str(frame_path))
        img_height, img_width = img.shape[:2]
        
        # 读取标注
        with open(label_path, 'r') as f:
            annotations = f.readlines()
        
        # 绘制每个标注
        for ann in annotations:
            class_id, x_center, y_center, width, height = map(float, ann.strip().split())
            
            # 转换回像素坐标
            x_center = int(x_center * img_width)
            y_center = int(y_center * img_height)
            width = int(width * img_width)
            height = int(height * img_height)
            
            # 计算边界框坐标
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签
            cv2.putText(img, f"Class {int(class_id)}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存可视化结果
        cv2.imwrite(str(vis_path), img)
    
    def process_video_and_annotations(self):
        """处理所有视频和标注"""
        # 获取所有视频文件
        video_files = list(self.video_dir.glob('*.mp4'))
        
        for video_path in video_files:
            print(f"处理视频: {video_path.name}")
            
            # 提取帧
            frame_count = self.extract_frames(video_path)
            
            # 查找对应的标注文件
            annotation_path = self.annotation_dir / f"{video_path.stem}.xml"
            if not annotation_path.exists():
                print(f"未找到标注文件: {annotation_path}")
                continue
            
            # 获取视频尺寸
            cap = cv2.VideoCapture(str(video_path))
            img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # 转换标注格式
            frame_annotations = self.convert_cvat_to_yolo(annotation_path, frame_count, img_width, img_height)
            
            # 处理每一帧
            frames_to_keep = []
            for frame_idx, annotations in frame_annotations.items():
                frame_path = self.frames_dir / f"{video_path.stem}_{frame_idx:06d}.jpg"
                label_path = self.labels_dir / f"{video_path.stem}_{frame_idx:06d}.txt"
                
                if annotations:  # 如果有标注
                    # 保存YOLO格式标注
                    with open(label_path, 'w') as f:
                        for ann in annotations:
                            f.write(f"{' '.join(map(str, ann))}\n")
                    frames_to_keep.append(frame_path)
                else:  # 如果没有标注，删除帧
                    if frame_path.exists():
                        frame_path.unlink()
            
            # 随机选择一些帧进行可视化
            if frames_to_keep:
                num_vis = min(5, len(frames_to_keep))  # 最多可视化5帧
                vis_frames = random.sample(frames_to_keep, num_vis)
                
                for frame_path in vis_frames:
                    frame_idx = int(frame_path.stem.split('_')[-1])
                    label_path = self.labels_dir / f"{video_path.stem}_{frame_idx:06d}.txt"
                    vis_path = self.vis_dir / f"{video_path.stem}_{frame_idx:06d}_vis.jpg"
                    
                    self.visualize_annotation(frame_path, label_path, vis_path)
            
            print(f"视频 {video_path.name} 处理完成")
            print(f"保留帧数: {len(frames_to_keep)}")
            print(f"删除帧数: {frame_count - len(frames_to_keep)}")
            print(f"可视化帧数: {num_vis if frames_to_keep else 0}")

def main():
    # 设置路径
    video_dir = "/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/annotation_videos"
    annotation_dir = "/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/annotations"
    output_dir = "/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/processed_data"
    
    # 创建处理器实例
    processor = AnnotationProcessor(video_dir, annotation_dir, output_dir)
    
    # 处理视频和标注
    processor.process_video_and_annotations()

if __name__ == "__main__":
    main() 