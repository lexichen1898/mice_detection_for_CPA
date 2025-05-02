import argparse
import subprocess
from datetime import datetime
import os

def time_to_seconds(time_str):
    """将时间字符串（格式：MM:SS）转换为秒数"""
    try:
        time_obj = datetime.strptime(time_str, '%M:%S')
        return time_obj.minute * 60 + time_obj.second
    except ValueError:
        raise ValueError("时间格式错误，请使用 MM:SS 格式（例如：3:40）")

def cut_video(input_path, start_time, end_time, output_path=None):
    """
    使用 ffmpeg 剪辑视频的指定时间段
    
    参数:
        input_path (str): 输入视频的路径
        start_time (str): 开始时间（格式：MM:SS）
        end_time (str): 结束时间（格式：MM:SS）
        output_path (str, optional): 输出视频的路径。如果不指定，将在原文件名后添加 _cut
    """
    try:
        # 转换时间为秒
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        
        if start_seconds >= end_seconds:
            raise ValueError("开始时间必须早于结束时间")
        
        # 如果没有指定输出路径，则生成默认路径
        if output_path is None:
            input_name = input_path.rsplit('.', 1)[0]
            output_path = f"{input_name}_cut.mp4"
        
        # 构建 ffmpeg 命令
        duration = end_seconds - start_seconds
        command = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_seconds),
            '-t', str(duration),
            '-c', 'copy',  # 直接复制流，不重新编码
            '-y',  # 覆盖已存在的文件
            output_path
        ]
        
        # 执行命令
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"视频剪辑完成！输出文件：{output_path}")
        else:
            print(f"处理视频时出错：{result.stderr}")
        
    except Exception as e:
        print(f"处理视频时出错：{str(e)}")

def main():
    parser = argparse.ArgumentParser(description='视频剪辑工具')
    parser.add_argument('input_path', help='输入视频的路径')
    parser.add_argument('start_time', help='开始时间（格式：MM:SS）')
    parser.add_argument('end_time', help='结束时间（格式：MM:SS）')
    parser.add_argument('--output', '-o', help='输出视频的路径（可选）')
    
    args = parser.parse_args()
    cut_video(args.input_path, args.start_time, args.end_time, args.output)

if __name__ == '__main__':
    main()

    #python src/video_cutter.py input_video.mp4 3:40 5:10 -o output_video.mp4