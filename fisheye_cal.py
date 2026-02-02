#!/usr/bin/env python3
"""
Fisheye Lens Calibration Tool
用于鱼眼镜头参数的矫正修复，支持实时调整参数并对比原视频和矫正后的视频
"""

import cv2
import numpy as np
import yaml
import argparse
import os
from pathlib import Path


class FisheyeCalibrationTool:
    def __init__(self, video_path, config_path, position, sample, start_frame=0):
        self.video_path = video_path
        self.config_path = config_path
        self.position = position
        self.sample = sample
        self.start_frame = start_frame
        
        # 加载配置
        self.config = self.load_config()
        
        # 获取或初始化鱼眼参数
        self.load_fisheye_params()
        
        # 视频捕获
        self.cap = None
        self.init_video()
        
        # 窗口名称
        self.original_window = 'Original Video (Fisheye)'
        self.undistorted_window = 'Undistorted Video'
        
        # 控制状态
        self.paused = False
        self.current_frame_num = start_frame
        
        # 参数调整步长
        self.step_size = {
            'fx': 10.0,
            'fy': 10.0,
            'cx': 5.0,
            'cy': 5.0,
            'k1': 0.001,
            'k2': 0.001,
            'k3': 0.001,
            'k4': 0.001
        }
        
        # 当前调整的参数
        self.current_param = 'fx'
        self.params_list = ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4']
        self.param_index = 0
        
    def load_config(self):
        """加载YAML配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            # 创建默认配置
            config = {
                'bev': {
                    'canvas_width': 800,
                    'canvas_height': 800,
                    'reference_rect': {
                        'aspect_ratio': 2.54,
                        'width_ratio': 0.4
                    }
                },
                'cameras': {
                    'front': {'file_name': '', 'offset_frame': 0},
                    'back': {'file_name': '', 'offset_frame': 0},
                    'left': {'file_name': '', 'offset_frame': 0},
                    'right': {'file_name': '', 'offset_frame': 0}
                }
            }
            self.save_config(config)
        
        return config
    
    def save_config(self, config=None):
        """保存配置到YAML文件"""
        if config is None:
            config = self.config
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    def load_fisheye_params(self):
        """加载鱼眼参数"""
        camera_config = self.config['cameras'].get(self.position, {})
        
        # 获取图像尺寸（从视频中读取第一帧来确定）
        # 先初始化一个默认值
        self.image_width = 1280
        self.image_height = 960
        
        # 加载K矩阵（内参矩阵）
        K = camera_config.get('fisheye_K', None)
        if K is None or len(K) != 3 or len(K[0]) != 3:
            # 初始化默认K矩阵（基于图像中心）
            fx = self.image_width * 0.7  # 默认焦距
            fy = self.image_height * 0.7
            cx = self.image_width / 2.0
            cy = self.image_height / 2.0
            self.K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
        else:
            self.K = np.array(K, dtype=np.float32)
            # 更新图像尺寸估计
            self.image_width = int(self.K[0, 2] * 2)
            self.image_height = int(self.K[1, 2] * 2)
        
        # 加载D矩阵（畸变系数）
        D = camera_config.get('fisheye_D', None)
        if D is None or len(D) != 4:
            # 初始化默认D矩阵（轻微鱼眼畸变）
            self.D = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            self.D = np.array(D, dtype=np.float32)
        
        # 计算新的相机矩阵（用于undistort）
        self.new_K = self.K.copy()
        self.map1 = None
        self.map2 = None
        self.update_maps()
    
    def init_video(self):
        """初始化视频捕获"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")
        
        # 设置起始帧
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.current_frame_num = self.start_frame
        
        # 获取视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 读取第一帧以获取图像尺寸
        ret, frame = self.cap.read()
        if ret:
            self.image_height, self.image_width = frame.shape[:2]
            # 更新K矩阵的中心点（如果使用默认值）
            if self.K[0, 2] == self.image_width / 2.0 or self.K[1, 2] == self.image_height / 2.0:
                self.K[0, 2] = self.image_width / 2.0
                self.K[1, 2] = self.image_height / 2.0
            self.update_maps()
            # 重置到起始帧
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
    
    def update_maps(self):
        """更新undistort映射表"""
        # 使用fisheye模型计算映射
        # estimateNewCameraMatrixForUndistortRectify 只返回一个值（新的相机矩阵）
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (self.image_width, self.image_height), np.eye(3), balance=0.0
        )
        
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K,
            (self.image_width, self.image_height), cv2.CV_16SC2
        )
    
    def undistort_frame(self, frame):
        """对帧进行鱼眼矫正"""
        if self.map1 is None or self.map2 is None:
            return frame
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    def draw_params_overlay(self, frame, is_original=True):
        """在帧上绘制参数信息"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # 创建半透明背景
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制参数信息
        y_offset = 30
        line_height = 25
        
        # 窗口标识
        window_name = "原始视频 (鱼眼)" if is_original else "矫正后视频"
        cv2.putText(frame, window_name, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height * 2
        
        # K矩阵参数
        cv2.putText(frame, f"fx: {self.K[0,0]:.2f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"fy: {self.K[1,1]:.2f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"cx: {self.K[0,2]:.2f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"cy: {self.K[1,2]:.2f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        
        # D矩阵参数
        cv2.putText(frame, f"k1: {self.D[0]:.6f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"k2: {self.D[1]:.6f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"k3: {self.D[2]:.6f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += line_height
        cv2.putText(frame, f"k4: {self.D[3]:.6f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 当前调整的参数（高亮显示）
        if not is_original:
            param_y = 30 + (self.param_index + 1) * line_height + 25
            if 30 <= param_y <= 200:
                cv2.putText(frame, "<--", (250, param_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 帧号信息
        cv2.putText(frame, f"Frame: {self.current_frame_num}/{self.total_frames}", 
                   (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 暂停状态
        if self.paused:
            cv2.putText(frame, "PAUSED", (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def adjust_param(self, param_name, direction):
        """调整参数"""
        step = self.step_size[param_name]
        if direction > 0:
            step = step
        else:
            step = -step
        
        if param_name == 'fx':
            self.K[0, 0] += step
        elif param_name == 'fy':
            self.K[1, 1] += step
        elif param_name == 'cx':
            self.K[0, 2] += step
        elif param_name == 'cy':
            self.K[1, 2] += step
        elif param_name == 'k1':
            self.D[0] += step
        elif param_name == 'k2':
            self.D[1] += step
        elif param_name == 'k3':
            self.D[2] += step
        elif param_name == 'k4':
            self.D[3] += step
        
        # 更新映射表
        self.update_maps()
        print(f"{param_name} = {self.get_param_value(param_name):.6f}")
    
    def get_param_value(self, param_name):
        """获取参数值"""
        if param_name == 'fx':
            return self.K[0, 0]
        elif param_name == 'fy':
            return self.K[1, 1]
        elif param_name == 'cx':
            return self.K[0, 2]
        elif param_name == 'cy':
            return self.K[1, 2]
        elif param_name == 'k1':
            return self.D[0]
        elif param_name == 'k2':
            return self.D[1]
        elif param_name == 'k3':
            return self.D[2]
        elif param_name == 'k4':
            return self.D[3]
        return 0.0
    
    def save_fisheye_params(self):
        """保存鱼眼参数到配置文件"""
        if self.position not in self.config['cameras']:
            self.config['cameras'][self.position] = {}
        
        camera_config = self.config['cameras'][self.position]
        camera_config['fisheye_K'] = self.K.tolist()
        camera_config['fisheye_D'] = self.D.tolist()
        
        self.save_config()
        print(f"\n✓ 鱼眼参数已保存到 {self.config_path}")
        print(f"K矩阵: {self.K}")
        print(f"D矩阵: {self.D}")
    
    def print_help(self):
        """打印帮助信息"""
        print("\n" + "="*60)
        print("鱼眼矫正工具 - 操作说明")
        print("="*60)
        print("参数调整:")
        print("  [1-8] 选择要调整的参数")
        print("    1: fx (焦距x)")
        print("    2: fy (焦距y)")
        print("    3: cx (主点x)")
        print("    4: cy (主点y)")
        print("    5: k1 (畸变系数1)")
        print("    6: k2 (畸变系数2)")
        print("    7: k3 (畸变系数3)")
        print("    8: k4 (畸变系数4)")
        print("  [↑/↓] 或 [+/-] 增加/减少当前参数值")
        print("  [←/→] 或 [a/d] 切换参数")
        print("\n视频控制:")
        print("  [空格] 或 [p] 暂停/继续播放")
        print("  [r] 重置到起始帧")
        print("  [s] 保存当前参数到配置文件")
        print("  [q] 退出")
        print("="*60)
    
    def run(self):
        """运行主循环"""
        self.print_help()
        
        # 创建窗口
        cv2.namedWindow(self.original_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.undistorted_window, cv2.WINDOW_NORMAL)
        
        # 调整窗口大小
        cv2.resizeWindow(self.original_window, 640, 480)
        cv2.resizeWindow(self.undistorted_window, 640, 480)
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    # 视频结束，循环播放
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                    self.current_frame_num = self.start_frame
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                
                self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 显示原始帧
            original_frame = self.draw_params_overlay(frame.copy(), is_original=True)
            cv2.imshow(self.original_window, original_frame)
            
            # 矫正并显示
            undistorted_frame = self.undistort_frame(frame.copy())
            undistorted_frame = self.draw_params_overlay(undistorted_frame, is_original=False)
            cv2.imshow(self.undistorted_window, undistorted_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') or key == ord('p'):
                self.paused = not self.paused
                print("暂停" if self.paused else "继续播放")
            elif key == ord('r'):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                self.current_frame_num = self.start_frame
                print(f"重置到帧 {self.start_frame}")
            elif key == ord('s'):
                self.save_fisheye_params()
            elif key >= ord('1') and key <= ord('8'):
                self.param_index = key - ord('1')
                self.current_param = self.params_list[self.param_index]
                print(f"选择参数: {self.current_param} = {self.get_param_value(self.current_param):.6f}")
            elif key == ord('+') or key == ord('=') or key == 82:  # +, =, 或上箭头
                self.adjust_param(self.current_param, 1)
            elif key == ord('-') or key == ord('_') or key == 84:  # -, _, 或下箭头
                self.adjust_param(self.current_param, -1)
            elif key == ord('a') or key == 81:  # a 或左箭头
                self.param_index = (self.param_index - 1) % len(self.params_list)
                self.current_param = self.params_list[self.param_index]
                print(f"切换到参数: {self.current_param} = {self.get_param_value(self.current_param):.6f}")
            elif key == ord('d') or key == 83:  # d 或右箭头
                self.param_index = (self.param_index + 1) % len(self.params_list)
                self.current_param = self.params_list[self.param_index]
                print(f"切换到参数: {self.current_param} = {self.get_param_value(self.current_param):.6f}")
        
        # 清理
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n程序退出")


def main():
    parser = argparse.ArgumentParser(description='鱼眼镜头矫正工具')
    parser.add_argument('--sample', type=int, required=True, help='样本编号 (例如: 1)')
    parser.add_argument('--position', type=str, required=True, 
                       choices=['front', 'back', 'left', 'right'],
                       help='摄像头位置')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--startfrom', type=int, default=0,
                       help='起始帧号 (默认: 0)')
    
    args = parser.parse_args()
    
    # 构建文件路径
    config_path = Path(args.config)
    data_dir = config_path.parent / 'data'
    video_file = f'sample{args.sample}_{args.position}.mp4'
    video_path = data_dir / video_file
    
    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    # 创建工具并运行
    tool = FisheyeCalibrationTool(
        video_path=str(video_path),
        config_path=str(config_path),
        position=args.position,
        sample=args.sample,
        start_frame=args.startfrom
    )
    
    tool.run()


if __name__ == '__main__':
    main()
