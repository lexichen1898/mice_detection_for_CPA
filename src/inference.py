from ultralytics import YOLO
import argparse
import cv2
import os
from pathlib import Path
import json
from datetime import datetime

def process_video(
    model_path,
    video_path,
    output_dir='runs/detect',
    conf_thres=0.25,
    save_video=True,
    save_json=True,
    output_name=None
):
    """
    处理视频并进行目标检测
    
    参数:
        model_path: YOLO模型路径
        video_path: 输入视频路径
        output_dir: 输出目录
        conf_thres: 置信度阈值
        save_video: 是否保存检测结果视频
        save_json: 是否保存检测结果JSON
        output_name: 输出文件名（不包含扩展名）
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 准备视频写入器
    video_writer = None
    if save_video:
        if output_name is None:
            output_name = Path(video_path).stem
        output_video_path = output_dir / f"{output_name}_detection.mp4"
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
    
    # 准备JSON结果
    detection_results = []
    frame_count = 0
    
    print(f"开始处理视频: {video_path}")
    print(f"总帧数: {total_frames}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 进行检测
        results = model(frame, conf=conf_thres)[0]
        
        # 在帧上绘制检测结果
        annotated_frame = results.plot()
        
        # 保存视频帧
        if save_video:
            video_writer.write(annotated_frame)
        
        # 保存检测结果
        if save_json:
            frame_detections = []
            for box in results.boxes:
                detection = {
                    'frame': frame_count,
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                frame_detections.append(detection)
            detection_results.extend(frame_detections)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧")
    
    # 释放资源
    cap.release()
    if video_writer:
        video_writer.release()
    
    # 保存JSON结果
    if save_json:
        if output_name is None:
            output_name = Path(video_path).stem
        json_path = output_dir / f"{output_name}_detection.json"
        with open(json_path, 'w') as f:
            json.dump(detection_results, f, indent=2)
    
    print(f"\n处理完成！")
    print(f"输出目录: {output_dir}")
    if save_video:
        print(f"检测结果视频: {output_video_path}")
    if save_json:
        print(f"检测结果JSON: {json_path}")
    
    return output_dir

def process_video_folder(
    model_path,
    video_folder,
    output_dir='runs/detect',
    conf_thres=0.25,
    save_video=True,
    save_json=True
):
    """
    处理文件夹中的所有视频文件
    
    参数:
        model_path: YOLO模型路径
        video_folder: 视频文件夹路径
        output_dir: 输出目录
        conf_thres: 置信度阈值
        save_video: 是否保存检测结果视频
        save_json: 是否保存检测结果JSON
    """
    video_folder = Path(video_folder)
    if not video_folder.exists():
        raise ValueError(f"视频文件夹不存在: {video_folder}")
    
    # 获取所有视频文件
    video_files = list(video_folder.glob("*.mp4")) + list(video_folder.glob("*.avi"))
    
    if not video_files:
        print(f"在文件夹 {video_folder} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    for video_path in video_files:
        print(f"\n处理视频: {video_path.name}")
        process_video(
            model_path=model_path,
            video_path=str(video_path),
            output_dir=output_dir,
            conf_thres=conf_thres,
            save_video=save_video,
            save_json=save_json,
            output_name=video_path.stem
        )

def main():
    parser = argparse.ArgumentParser(description='使用YOLO模型进行视频检测')
    parser.add_argument('--model', type=str, default='/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/runs/train/mice_detector/weights/best.pt', help='YOLO模型路径')
    parser.add_argument('--video', type=str, help='输入视频路径或视频文件夹路径')
    parser.add_argument('--output-dir', type=str, default='runs/detect', help='输出目录')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--no-save-video', action='store_true', help='不保存检测结果视频')
    parser.add_argument('--no-save-json', action='store_true', help='不保存检测结果JSON')
    
    args = parser.parse_args()
    
    if not args.video:
        raise ValueError("必须指定视频路径或视频文件夹路径")
    
    video_path = Path(args.video)
    if video_path.is_dir():
        process_video_folder(
            model_path=args.model,
            video_folder=args.video,
            output_dir=args.output_dir,
            conf_thres=args.conf_thres,
            save_video=not args.no_save_video,
            save_json=not args.no_save_json
        )
    else:
        process_video(
            model_path=args.model,
            video_path=args.video,
            output_dir=args.output_dir,
            conf_thres=args.conf_thres,
            save_video=not args.no_save_video,
            save_json=not args.no_save_json
        )

if __name__ == '__main__':
    main() 