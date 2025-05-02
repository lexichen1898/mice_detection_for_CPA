<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        .language-switch {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }
        .language-switch button {
            padding: 5px 10px;
            margin: 0 5px;
            cursor: pointer;
        }
        .active {
            background-color: #007bff;
            color: white;
        }
        .content {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="language-switch">
        <button onclick="switchLanguage('en')" id="enBtn" class="active">English</button>
        <button onclick="switchLanguage('zh')" id="zhBtn">中文</button>
    </div>

    <div class="content">
        <div id="en">
# Mice Detection Project

A computer vision project for mouse behavior detection using YOLO model.

## Environment Setup

This project uses Conda for environment management. Follow these steps to set up the environment:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)

2. Create and activate the environment:
```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate mice_detection
```

## Project Structure

```
mice_detection/
├── src/                    # Source code directory
│   ├── inference.py        # Inference script
│   ├── train_yolo.py       # Training script
│   ├── generate_heatmap.py # Heatmap generation
│   ├── mark_sections_web.py # Region marking web interface
│   ├── process_annotations.py # Annotation processing
│   ├── split_dataset.py    # Dataset splitting
│   ├── video_cutter.py     # Video cutting
│   └── templates/          # Web interface templates
├── videos/                 # Directory for input videos
├── processed_data/         # Directory for processed data
│   ├── annotations/        # Annotation files
│   └── labels/            # Label files
├── runs/                  # Directory for model outputs
│   ├── detect/           # Detection results
│   ├── train/            # Training results
│   └── heatmaps/         # Heatmap results
├── environment.yml        # Environment configuration
└── README.md             # Project documentation
```

## Main Features

1. Video object detection
2. Heatmap generation
3. Region marking
4. Dataset processing
5. Model training

## Usage

### 1. Video Detection
```bash
python src/inference.py --video <video_path> --model <model_path>
```

### 2. Model Training
```bash
python src/train_yolo.py --data <dataset_config> --epochs <num_epochs>
```

### 3. Generate Heatmap
```bash
python src/generate_heatmap.py --video <video_path> --json <detection_json>
```

### 4. Start Region Marking Web Interface
```bash
python src/mark_sections_web.py
```

## Notes

1. Ensure all dependencies are correctly installed
2. Make sure you have sufficient disk space before use
3. GPU is recommended for training and inference
4. Video file paths may contain spaces, please wrap them in quotes
        </div>

        <div id="zh" style="display: none;">
# Lexi Mice Detection

## 项目概述
这是一个用于检测小鼠行为的计算机视觉项目，使用 YOLO 模型进行目标检测。

## 环境配置
本项目使用 Conda 管理环境。请按照以下步骤配置环境：

1. 安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/download)

2. 创建并激活环境：
```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate lexi_mice_detection
```

## 项目结构
```
lexi_mice_detection/
├── src/                    # 源代码目录
│   ├── inference.py        # 推理脚本
│   ├── train_yolo.py       # 训练脚本
│   ├── generate_heatmap.py # 热力图生成
│   ├── mark_sections_web.py # 区域标记Web界面
│   ├── process_annotations.py # 标注处理
│   ├── split_dataset.py    # 数据集分割
│   ├── video_cutter.py     # 视频切割
│   └── templates/          # Web界面模板
├── videos/                 # 输入视频目录
├── processed_data/         # 处理后的数据目录
│   ├── annotations/        # 标注文件
│   └── labels/            # 标签文件
├── runs/                  # 模型输出目录
│   ├── detect/           # 检测结果
│   ├── train/            # 训练结果
│   └── heatmaps/         # 热力图结果
├── environment.yml        # 环境配置文件
└── README.md              # 项目说明文档
```

## 主要功能
1. 视频目标检测
2. 热力图生成
3. 区域标记
4. 数据集处理
5. 模型训练

## 使用方法
### 1. 视频检测
```bash
python src/inference.py --video <视频路径> --model <模型路径>
```

### 2. 训练模型
```bash
python src/train_yolo.py --data <数据集配置文件> --epochs <训练轮数>
```

### 3. 生成热力图
```bash
python src/generate_heatmap.py --video <视频路径> --json <检测结果JSON>
```

### 4. 启动区域标记Web界面
```bash
python src/mark_sections_web.py
```

## 注意事项
1. 确保已正确安装所有依赖
2. 使用前请确保有足够的磁盘空间
3. 建议使用GPU进行训练和推理
4. 视频文件路径中可能包含空格，请使用引号包裹路径
        </div>
    </div>

    <script>
        function switchLanguage(lang) {
            document.getElementById('en').style.display = lang === 'en' ? 'block' : 'none';
            document.getElementById('zh').style.display = lang === 'zh' ? 'block' : 'none';
            
            document.getElementById('enBtn').classList.toggle('active', lang === 'en');
            document.getElementById('zhBtn').classList.toggle('active', lang === 'zh');
        }
    </script>
</body>
</html> 