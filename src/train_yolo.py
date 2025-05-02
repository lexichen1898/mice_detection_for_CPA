from ultralytics import YOLO
import argparse
import yaml
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(
    data_yaml,
    model_size='n',  # n, s, m, l, x
    epochs=100,
    batch_size=16,
    img_size=640,
    device='0',  # 使用GPU，如果使用CPU则设为'cpu'
    config_path=None
):
    """
    使用YOLO训练模型
    
    参数:
        data_yaml: 数据集配置文件路径
        model_size: 模型大小 (n, s, m, l, x)
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像大小
        device: 训练设备
        config_path: 额外配置文件路径
    """
    # 加载配置
    config = {}
    if config_path:
        config = load_config(config_path)
    
    # 加载预训练模型
    model = YOLO(f'yolov8{model_size}.pt')  # 这将自动下载预训练模型
    
    # 设置训练参数
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'patience': 50,  # 早停耐心值
        'save': True,  # 保存结果
        'project': 'runs/train',  # 保存目录
        'name': 'mice_detector',  # 实验名称
        'exist_ok': True,  # 允许覆盖已存在的实验目录
        'pretrained': True,  # 使用预训练权重
        'optimizer': 'auto',  # 优化器
        'lr0': 0.01,  # 初始学习率
        'lrf': 0.01,  # 最终学习率
        'momentum': 0.937,  # SGD动量
        'weight_decay': 0.0005,  # 权重衰减
        'warmup_epochs': 3,  # 预热轮数
        'warmup_momentum': 0.8,  # 预热动量
        'warmup_bias_lr': 0.1,  # 预热偏置学习率
        'box': 7.5,  # 框损失增益
        'cls': 0.5,  # 分类损失增益
        'dfl': 1.5,  # DFL损失增益
        'pose': 12.0,  # 姿态损失增益
        'kobj': 1.0,  # 关键点目标损失增益
        'label_smoothing': 0.0,  # 标签平滑
        'nbs': 64,  # 标称批次大小
        'overlap_mask': True,  # 重叠掩码
        'mask_ratio': 4,  # 掩码下采样率
        'dropout': 0.0,  # 使用dropout正则化
        'val': True,  # 验证
    }
    
    # 更新配置
    train_args.update(config)
    
    # 开始训练
    results = model.train(**train_args)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='训练YOLO模型')
    parser.add_argument('--data', type=str, default='src/dataset.yaml', help='数据集配置文件路径')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='图像大小')
    parser.add_argument('--device', type=str, default='0', help='训练设备')
    parser.add_argument('--config', type=str, help='额外配置文件路径')
    
    args = parser.parse_args()
    
    # 开始训练
    results = train_model(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        config_path=args.config
    )
    
    print("训练完成！")
    print(f"结果保存在: {results.save_dir}")

if __name__ == '__main__':
    main() 