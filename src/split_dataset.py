import os
import shutil
from pathlib import Path
import random

def split_dataset(
    processed_data_dir,
    output_dir,
    val_ratio=0.2,
    random_seed=42
):
    """
    将处理后的数据集分割为训练集和验证集
    
    参数:
        processed_data_dir: 处理后的数据目录
        output_dir: 输出目录
        val_ratio: 验证集比例
        random_seed: 随机种子
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 创建目录
    processed_data_dir = Path(processed_data_dir)
    output_dir = Path(output_dir)
    
    # 创建训练集和验证集目录
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    
    for dir_path in [train_dir, val_dir]:
        (dir_path / 'images').mkdir(parents=True, exist_ok=True)
        (dir_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    frames_dir = processed_data_dir / 'frames'
    labels_dir = processed_data_dir / 'labels'
    
    image_files = list(frames_dir.glob('*.jpg'))
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 计算验证集大小
    val_size = int(len(image_files) * val_ratio)
    
    # 分割数据集
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]
    
    # 复制文件
    for files, target_dir in [(train_files, train_dir), (val_files, val_dir)]:
        for img_path in files:
            # 获取对应的标签文件
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # 复制图像
            shutil.copy2(img_path, target_dir / 'images' / img_path.name)
            
            # 复制标签
            if label_path.exists():
                shutil.copy2(label_path, target_dir / 'labels' / label_path.name)
    
    # 打印统计信息
    print(f"数据集分割完成！")
    print(f"总图像数: {len(image_files)}")
    print(f"训练集图像数: {len(train_files)}")
    print(f"验证集图像数: {len(val_files)}")
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")

def main():
    # 设置路径
    processed_data_dir = "/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/processed_data"
    output_dir = "/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/dataset"
    
    # 分割数据集
    split_dataset(processed_data_dir, output_dir)

if __name__ == '__main__':
    main() 