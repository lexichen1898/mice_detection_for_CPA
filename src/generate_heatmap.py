import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict

class HeatmapGenerator:
    def __init__(self, video_path, sections_path, detections_path):
        """初始化热图生成器
        
        Args:
            video_path: 视频文件路径
            sections_path: 区域标记JSON文件路径
            detections_path: 检测结果JSON文件路径
        """
        self.video_path = Path(video_path)
        self.sections_path = Path(sections_path)
        self.detections_path = Path(detections_path)
        
        # 加载区域标记和检测结果
        with open(sections_path, 'r') as f:
            self.sections = json.load(f)['sections']
            
        with open(detections_path, 'r') as f:
            self.detections = json.load(f)
            
        # 初始化热图字典
        self.heatmaps = {}
        
        # 初始化统计信息
        self.stats = {
            'section_stats': defaultdict(lambda: {
                'total_time': 0,
                'visit_count': 0,
                'last_frame': -1,
                'last_mouse_id': None,
                'subsections': {}
            }),
            'frame_stats': defaultdict(list)  # 记录每帧中每个区域的小鼠信息
        }
        
        # 为每个区域初始化子区域统计
        for section in self.sections:
            section_id = section['id']
            if section['separator_type'] == 'vertical':
                self.stats['section_stats'][section_id]['subsections'] = {
                    'left': {'total_time': 0, 'visit_count': 0},
                    'right': {'total_time': 0, 'visit_count': 0}
                }
            elif section['separator_type'] == 'horizontal':
                self.stats['section_stats'][section_id]['subsections'] = {
                    'top': {'total_time': 0, 'visit_count': 0},
                    'bottom': {'total_time': 0, 'visit_count': 0}
                }
            else:
                self.stats['section_stats'][section_id]['subsections'] = {
                    'first': {'total_time': 0, 'visit_count': 0},
                    'second': {'total_time': 0, 'visit_count': 0}
                }

    def _get_subsection(self, section, x, y):
        """根据坐标确定子区域"""
        if not section['separator']:
            return 'first'
            
        sep = section['separator']
        if section['separator_type'] == 'vertical':
            return 'left' if x < sep['x1'] else 'right'
        else:  # horizontal
            return 'top' if y < sep['y1'] else 'bottom'

    def process_detections(self):
        """处理检测结果并生成热图"""
        # 获取视频尺寸
        if self.video_path.exists():
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            # 如果没有视频文件，从检测结果中获取最大尺寸
            width = height = 0
            for detection in self.detections:
                bbox = detection['bbox']
                width = max(width, int(bbox[2]))
                height = max(height, int(bbox[3]))
        
        # 为每个区域初始化热图
        for section in self.sections:
            self.heatmaps[section['id']] = np.zeros((height, width), dtype=np.float32)
        
        # 处理每个检测结果
        print("正在处理检测结果...")
        for detection in tqdm(self.detections):
            frame = detection['frame']
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            confidence = detection['confidence']
            
            # 确保边界框坐标是整数
            bbox = [int(x) for x in bbox]
            
            # 计算边界框中心点
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # 检查边界框是否在区域内，并更新相应的热图
            for section in self.sections:
                coords = section['coords']
                # 确保区域坐标是整数
                coords = {k: int(v) for k, v in coords.items()}
                
                # 检查中心点是否在区域内
                if (coords['x1'] <= center_x <= coords['x2'] and 
                    coords['y1'] <= center_y <= coords['y2']):
                    
                    # 确定子区域
                    subsection = self._get_subsection(section, center_x, center_y)
                    
                    # 更新该帧的区域统计
                    self.stats['frame_stats'][frame].append({
                        'section_id': section['id'],
                        'subsection': subsection,
                        'confidence': confidence,
                        'mouse_id': detection.get('mouse_id', None)
                    })
                    
                    # 更新热图
                    x1 = max(bbox[0], coords['x1'])
                    y1 = max(bbox[1], coords['y1'])
                    x2 = min(bbox[2], coords['x2'])
                    y2 = min(bbox[3], coords['y2'])
                    
                    if x2 > x1 and y2 > y1:
                        self.heatmaps[section['id']][y1:y2, x1:x2] += confidence
        
        # 处理统计信息
        self._process_statistics()
        
    def _process_statistics(self):
        """处理统计信息，计算每个区域的停留时间和访问次数"""
        # 按帧顺序处理
        frames = sorted(self.stats['frame_stats'].keys())
        for frame in frames:
            frame_detections = self.stats['frame_stats'][frame]
            
            # 对每个区域，选择置信度最高的小鼠
            section_mice = {}
            for det in frame_detections:
                section_id = det['section_id']
                if (section_id not in section_mice or 
                    det['confidence'] > section_mice[section_id]['confidence']):
                    section_mice[section_id] = det
            
            # 更新统计信息
            for section_id, mouse_info in section_mice.items():
                stats = self.stats['section_stats'][section_id]
                
                # 如果是新的访问（与上一帧不同的小鼠或间隔超过1帧）
                if (mouse_info['mouse_id'] != stats['last_mouse_id'] or 
                    frame - stats['last_frame'] > 1):
                    stats['visit_count'] += 1
                    stats['subsections'][mouse_info['subsection']]['visit_count'] += 1
                
                # 更新停留时间
                stats['total_time'] += 1
                stats['subsections'][mouse_info['subsection']]['total_time'] += 1
                stats['last_frame'] = frame
                stats['last_mouse_id'] = mouse_info['mouse_id']
    
    def save_statistics(self, output_dir):
        """保存统计信息到文本文件
        
        Args:
            output_dir: 输出目录路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats_path = output_dir / 'section_statistics.txt'
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("区域统计信息\n")
            f.write("=" * 50 + "\n\n")
            
            for section in self.sections:
                section_id = section['id']
                stats = self.stats['section_stats'][section_id]
                
                f.write(f"区域 {section_id}:\n")
                f.write(f"  总停留时间: {stats['total_time']} 帧\n")
                f.write(f"  访问次数: {stats['visit_count']} 次\n")
                f.write(f"  平均每次停留时间: {stats['total_time']/max(1, stats['visit_count']):.2f} 帧\n")
                
                # 添加子区域统计
                f.write("\n  子区域统计:\n")
                for subsection, sub_stats in stats['subsections'].items():
                    subsection_name = section.get('name_tags', {}).get(subsection, {}).get('name', subsection)
                    f.write(f"    {subsection_name}:\n")
                    f.write(f"      停留时间: {sub_stats['total_time']} 帧\n")
                    f.write(f"      访问次数: {sub_stats['visit_count']} 次\n")
                    f.write(f"      平均每次停留时间: {sub_stats['total_time']/max(1, sub_stats['visit_count']):.2f} 帧\n")
                
                f.write("\n")
        
        print(f"统计信息已保存到: {stats_path}")
        
    def normalize_heatmaps(self):
        """归一化所有热图"""
        if not self.heatmaps:
            raise ValueError("请先处理检测结果")
            
        print("正在归一化热图...")
        for section_id in self.heatmaps:
            # 归一化到0-255
            self.heatmaps[section_id] = cv2.normalize(self.heatmaps[section_id], None, 0, 255, cv2.NORM_MINMAX)
            self.heatmaps[section_id] = self.heatmaps[section_id].astype(np.uint8)
        
    def apply_colormap(self):
        """应用颜色映射到所有热图"""
        if not self.heatmaps:
            raise ValueError("请先处理检测结果")
            
        print("正在应用颜色映射...")
        for section_id in self.heatmaps:
            # 应用JET颜色映射
            self.heatmaps[section_id] = cv2.applyColorMap(self.heatmaps[section_id], cv2.COLORMAP_JET)
            
    def draw_section_borders(self):
        """在所有热图上绘制区域边界和子区域"""
        if not self.heatmaps:
            raise ValueError("请先处理检测结果")
            
        print("正在绘制区域边界和子区域...")
        for section_id, heatmap in self.heatmaps.items():
            # 找到对应的区域
            section = next(s for s in self.sections if s['id'] == section_id)
            coords = section['coords']
            # 确保坐标是整数
            coords = {k: int(v) for k, v in coords.items()}
            
            # 绘制主区域边界
            cv2.rectangle(
                heatmap,
                (coords['x1'], coords['y1']),
                (coords['x2'], coords['y2']),
                (255, 255, 255),  # 白色边界
                2  # 线宽
            )
            
            # 绘制分隔线
            if section['separator']:
                sep = section['separator']
                cv2.line(
                    heatmap,
                    (sep['x1'], sep['y1']),
                    (sep['x2'], sep['y2']),
                    (255, 255, 255),  # 白色分隔线
                    2  # 线宽
                )
                
                # 绘制子区域名称
                if 'name_tags' in section:
                    for subsection, tag in section['name_tags'].items():
                        # 计算名称显示位置（在子区域外部）
                        if section['separator_type'] == 'vertical':
                            if subsection == 'left':
                                text_x = coords['x1'] - 10
                                text_y = coords['y1'] + 30
                            else:  # right
                                text_x = coords['x2'] + 10
                                text_y = coords['y1'] + 30
                        else:  # horizontal
                            if subsection == 'top':
                                text_x = coords['x1'] + 10
                                text_y = coords['y1'] - 10
                            else:  # bottom
                                text_x = coords['x1'] + 10
                                text_y = coords['y2'] + 30
                        
                        # 绘制名称背景（半透明黑色）
                        text = tag['name']
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # 绘制背景矩形
                        cv2.rectangle(
                            heatmap,
                            (text_x - 5, text_y - text_height - 5),
                            (text_x + text_width + 5, text_y + 5),
                            (0, 0, 0),
                            -1  # 填充矩形
                        )
                        
                        # 绘制名称文本
                        cv2.putText(
                            heatmap,
                            text,
                            (text_x, text_y),
                            font,
                            font_scale,
                            (255, 255, 255),  # 白色文字
                            thickness
                        )
            
            # 添加区域ID标签
            cv2.putText(
                heatmap,
                f'Section {section_id}',
                (coords['x1'], coords['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # 白色文字
                2
            )
        
    def save_heatmaps(self, output_dir):
        """保存所有热图
        
        Args:
            output_dir: 输出目录路径
        """
        if not self.heatmaps:
            raise ValueError("请先处理检测结果")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"正在保存热图到 {output_dir}...")
        for section_id, heatmap in self.heatmaps.items():
            output_path = output_dir / f'section_{section_id}_heatmap.jpg'
            cv2.imwrite(str(output_path), heatmap)
            
    def overlay_heatmap(self, frame, section_id, alpha=0.5):
        """将指定区域的热图叠加到原始帧上
        
        Args:
            frame: 原始帧
            section_id: 区域ID
            alpha: 透明度
            
        Returns:
            叠加后的图像
        """
        if not self.heatmaps or section_id not in self.heatmaps:
            raise ValueError(f"区域 {section_id} 的热图不存在")
            
        # 确保热图和帧大小相同
        heatmap_resized = cv2.resize(self.heatmaps[section_id], (frame.shape[1], frame.shape[0]))
        
        # 叠加热图
        overlay = cv2.addWeighted(frame, 1-alpha, heatmap_resized, alpha, 0)
        return overlay

def main():
    parser = argparse.ArgumentParser(description='从检测结果生成热图')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--sections', type=str, required=True, help='区域标记JSON文件路径')
    parser.add_argument('--detections', type=str, required=True, help='检测结果JSON文件路径')
    parser.add_argument('--output-dir', type=str, default='heatmaps', help='输出目录路径')
    parser.add_argument('--overlay-dir', type=str, help='输出叠加热图的视频目录路径（可选）')
    
    args = parser.parse_args()
    
    # 创建热图生成器
    generator = HeatmapGenerator(args.video, args.sections, args.detections)
    
    # 处理检测结果
    generator.process_detections()
    
    # 归一化热图
    generator.normalize_heatmaps()
    
    # 应用颜色映射
    generator.apply_colormap()
    
    # 绘制区域边界
    generator.draw_section_borders()
    
    # 保存热图和统计信息
    generator.save_heatmaps(args.output_dir)
    generator.save_statistics(args.output_dir)
    
    # 如果需要生成叠加视频
    if args.overlay_dir and args.video:
        print("正在生成叠加视频...")
        overlay_dir = Path(args.overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(args.video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 为每个区域生成叠加视频
        for section_id in generator.heatmaps:
            output_path = overlay_dir / f'section_{section_id}_overlay.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到开始
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 叠加热图
                overlay = generator.overlay_heatmap(frame, section_id)
                out.write(overlay)
                
            out.release()
            print(f"区域 {section_id} 的叠加视频已保存到 {output_path}")
            
        cap.release()
    
    print("处理完成！")

if __name__ == '__main__':
    main() 

    # python src/generate_heatmap.py --video "/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/videos/CPA_Post_Videos/Trial     1.mp4" --sections "/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/post_trail1_sections.json" --detections "/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/runs/detect/Post_detects/Trial     1_detection.json" --output-dir "/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/runs/heatmaps/post_trial1_heatmaps"