from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
import base64
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw
import socket
import math

app = Flask(__name__)

class SectionMarker:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.original = self.image.copy()
        self.sections = []
        self.current_section = None
        self.current_line = None
        self.current_section_id = None
        # 预定义的子区域名称
        self.subsection_names = {
            'vertical': {
                'left': '左侧区域',
                'right': '右侧区域'
            },
            'horizontal': {
                'top': '上方区域',
                'bottom': '下方区域'
            }
        }
        
    def add_section(self, x1, y1, x2, y2):
        if len(self.sections) < 4:
            self.sections.append({
                'id': len(self.sections) + 1,
                'coords': {
                    'x1': min(x1, x2),
                    'y1': min(y1, y2),
                    'x2': max(x1, x2),
                    'y2': max(y1, y2)
                },
                'separator': None,  # 分隔线信息
                'subsections': {    # 子区域信息
                    'first': None,
                    'second': None
                },
                'separator_type': None,  # 'vertical' 或 'horizontal'
                'name_tags': {}  # 存储名称标签
            })
            return True
        return False
    
    def add_separator(self, section_id, x1, y1, x2, y2):
        """为指定区域添加分隔线，并自动判断分割方向"""
        for section in self.sections:
            if section['id'] == section_id:
                # 获取区域边界
                sx1, sy1 = section['coords']['x1'], section['coords']['y1']
                sx2, sy2 = section['coords']['x2'], section['coords']['y2']
                
                # 确保分隔线在区域内
                x1 = max(sx1, min(sx2, x1))
                x2 = max(sx1, min(sx2, x2))
                y1 = max(sy1, min(sy2, y1))
                y2 = max(sy1, min(sy2, y2))
                
                # 计算分隔线角度
                dx = x2 - x1
                dy = y2 - y1
                angle = abs(math.degrees(math.atan2(dy, dx)))
                
                # 判断分割方向
                if angle < 45:  # 更接近水平
                    section['separator_type'] = 'horizontal'
                    # 调整到水平线
                    y = (y1 + y2) // 2
                    x1, x2 = sx1, sx2
                    y1 = y2 = y
                else:  # 更接近垂直
                    section['separator_type'] = 'vertical'
                    # 调整到垂直线
                    x = (x1 + x2) // 2
                    y1, y2 = sy1, sy2
                    x1 = x2 = x
                
                section['separator'] = {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2)
                }
                
                return True
        return False
    
    def add_name_tag(self, section_id, subsection, name):
        """添加名称标签到指定区域"""
        for section in self.sections:
            if section['id'] == section_id:
                if 'name_tags' not in section:
                    section['name_tags'] = {}
                section['name_tags'][subsection] = {
                    'name': name
                }
                return True
        return False
    
    def delete_section(self, section_id):
        """删除指定ID的区域"""
        self.sections = [s for s in self.sections if s['id'] != section_id]
        # 重新编号
        for i, section in enumerate(self.sections):
            section['id'] = i + 1
        return True
    
    def get_image_with_sections(self):
        img = self.original.copy()
        for section in self.sections:
            x1, y1 = section['coords']['x1'], section['coords']['y1']
            x2, y2 = section['coords']['x2'], section['coords']['y2']
            
            # 绘制区域边界
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制区域ID
            section_text = f'Section {section["id"]}'
            cv2.putText(img, section_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制分隔线
            if section['separator']:
                sep = section['separator']
                cv2.line(img, (sep['x1'], sep['y1']), (sep['x2'], sep['y2']), (255, 0, 0), 2)
            
            # 绘制名称标签
            if 'name_tags' in section:
                for subsection, tag in section['name_tags'].items():
                    # 计算标签显示位置
                    if section['separator_type'] == 'vertical':
                        if subsection == 'left':
                            text_x = x1 + 10
                            text_y = (y1 + y2) // 2
                        else:  # right
                            text_x = x2 - 100
                            text_y = (y1 + y2) // 2
                    else:  # horizontal
                        if subsection == 'top':
                            text_x = (x1 + x2) // 2
                            text_y = y1 + 30
                        else:  # bottom
                            text_x = (x1 + x2) // 2
                            text_y = y2 - 10
                    
                    # 使用系统默认字体
                    fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux系统默认字体
                    try:
                        font = ImageFont.truetype(fontpath, 32)
                    except:
                        # 如果找不到指定字体，使用默认字体
                        font = ImageFont.load_default()
                    
                    img_pil = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((text_x, text_y), tag['name'], font=font, fill=(255, 0, 0))
                    img = np.array(img_pil)
        
        return img
    
    def save_sections(self, output_path):
        if len(self.sections) != 4:
            return False
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sections_data = {
            'sections': self.sections
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sections_data, f, indent=2, ensure_ascii=False)
        
        return True

# 全局变量存储当前标记器实例和输出路径
marker = None
output_path = None

@app.route('/')
def index():
    return render_template('mark_sections.html')

@app.route('/get_image')
def get_image():
    if marker is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    img = marker.get_image_with_sections()
    img = Image.fromarray(img)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({
        'image': img_str,
        'sections_count': len(marker.sections),
        'sections': marker.sections
    })

@app.route('/add_section', methods=['POST'])
def add_section():
    if marker is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    data = request.json
    success = marker.add_section(
        int(data['x1']), int(data['y1']),
        int(data['x2']), int(data['y2'])
    )
    
    return jsonify({
        'success': success,
        'sections_count': len(marker.sections),
        'sections': marker.sections
    })

@app.route('/delete_section', methods=['POST'])
def delete_section():
    if marker is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    data = request.json
    success = marker.delete_section(int(data['section_id']))
    
    return jsonify({
        'success': success,
        'sections_count': len(marker.sections),
        'sections': marker.sections
    })

@app.route('/add_separator', methods=['POST'])
def add_separator():
    if marker is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    data = request.json
    success = marker.add_separator(
        int(data['section_id']),
        data['x1'],
        data['y1'],
        data['x2'],
        data['y2']
    )
    
    return jsonify({
        'success': success,
        'sections_count': len(marker.sections),
        'sections': marker.sections
    })

@app.route('/add_name_tag', methods=['POST'])
def add_name_tag():
    if marker is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    data = request.json
    success = marker.add_name_tag(
        int(data['section_id']),
        data['subsection'],
        data['name']
    )
    
    return jsonify({
        'success': success,
        'sections_count': len(marker.sections),
        'sections': marker.sections
    })

def extract_first_frame(video_path, output_path):
    """从视频中提取第一帧"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(output_path, frame)
        return True
    return False
    
@app.route('/save_sections', methods=['POST'])
def save_sections():
    if marker is None:
        return jsonify({'error': 'No image loaded'}), 400
    
    if output_path is None:
        return jsonify({'error': 'Output path not set'}), 400
    
    success = marker.save_sections(output_path)
    
    return jsonify({
        'success': success,
        'message': f'区域标记已保存到: {output_path}' if success else '保存失败'
    })

def find_available_port(start_port=5000, max_port=5100):
    """查找可用端口"""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError('没有找到可用的端口')

def main():
    parser = argparse.ArgumentParser(description='标记视频中的四个区域（网页版）')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--port', type=int, default=5000, help='网页服务器端口（可选，默认自动查找可用端口）')
    parser.add_argument('--output', type=str, default='sections.json', help='输出区域标记JSON文件名')
    
    args = parser.parse_args()
    
    # 提取第一帧
    frame_path = Path(args.video).parent / 'first_frame.jpg'
    if not extract_first_frame(args.video, str(frame_path)):
        print("无法从视频中提取第一帧")
        return
    
    # 创建HTML模板
    # create_html_template()
    
    # 初始化标记器和输出路径
    global marker, output_path
    marker = SectionMarker(str(frame_path))
    output_path = Path(args.video).parent / args.output
    
    # 查找可用端口
    try:
        port = args.port if args.port != 5000 else find_available_port()
        print(f"请在浏览器中访问: http://localhost:{port}")
        print(f"区域标记将保存到: {output_path}")
        app.run(host='0.0.0.0', port=port)
    except RuntimeError as e:
        print(f"错误: {e}")
        return

if __name__ == '__main__':
    main() 

    # python src/mark_sections_web.py --video "/home/zzmonlyyou/Documents/Projects/lexi_mice_detection/videos/CPA_Post_Videos/Trial     1.mp4" --output "post_trial1_sections.json"