#!/usr/bin/env python3
"""
BEV Homography Matrix Tool
用于创建和编辑BEV（Bird Eye View）的单应性矩阵H
"""

import cv2
import numpy as np
import yaml
import argparse
import os
from pathlib import Path


class BEVHomographyTool:
    def __init__(self, video_path, config_path, position, sample, num_lines=6, cal_mode=False, start_frame=0):
        self.video_path = video_path
        self.config_path = config_path
        self.position = position
        self.sample = sample
        self.num_lines = num_lines  # 横纵辅助线的数量
        self.cal_mode = cal_mode  # 校准模式：保持BEV ROI不变，只调整摄像头ROI
        self.start_frame = start_frame  # 起始帧号
        
        # 加载配置
        self.config = self.load_config()
        
        # 获取BEV画布尺寸
        self.bev_width = self.config['bev']['canvas_width']
        self.bev_height = self.config['bev']['canvas_height']
        
        # 获取参考矩形参数
        ref_rect = self.config['bev'].get('reference_rect', {})
        self.ref_aspect_ratio = ref_rect.get('aspect_ratio', 2.54)  # 长宽比
        self.ref_width_ratio = ref_rect.get('width_ratio', 0.4)  # 宽度占画布的比例
        
        # 点选择状态
        self.camera_points = []
        self.bev_points = []
        self.current_mode = 'camera'  # 'camera' 或 'bev'
        self.point_count = 0
        self.closed_loop_camera = False  # 摄像头视图是否已闭合
        self.closed_loop_bev = False  # BEV画布是否已闭合
        self.active_window = None  # 当前活动的窗口（用于k键重置）
        
        # 窗口名称
        self.camera_window = 'Camera View - Click points for ROI'
        self.bev_window = 'BEV Canvas - Click corresponding points'
        
    def load_config(self):
        """加载YAML配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 如果配置文件不存在，返回默认配置
            return {
                'bev': {
                    'canvas_width': 800, 
                    'canvas_height': 800,
                    'reference_rect': {
                        'aspect_ratio': 2.54,
                        'width_ratio': 0.4
                    }
                },
                'cameras': {
                    'front': {'file_name': '', 'offset_frame': 0, 'src_roi': [], 'dst_roi': [], 'H': []},
                    'back': {'file_name': '', 'offset_frame': 0, 'src_roi': [], 'dst_roi': [], 'H': []},
                    'left': {'file_name': '', 'offset_frame': 0, 'src_roi': [], 'dst_roi': [], 'H': []},
                    'right': {'file_name': '', 'offset_frame': 0, 'src_roi': [], 'dst_roi': [], 'H': []}
                }
            }
    
    def save_config(self):
        """保存配置到YAML文件"""
        # 更新当前摄像机的配置（保留原有的offset_frame等参数）
        camera_config = self.config['cameras'][self.position]
        camera_config['file_name'] = f'sample{self.sample}_{self.position}.mp4'
        # 保留原有的offset_frame，如果不存在则设为0
        if 'offset_frame' not in camera_config:
            camera_config['offset_frame'] = 0
        
        # 保存点坐标（确保是list格式，如果为空则清空旧数据）
        if len(self.camera_points) > 0:
            camera_config['src_roi'] = [[int(pt[0]), int(pt[1])] for pt in self.camera_points]
        else:
            camera_config['src_roi'] = []
        
        if len(self.bev_points) > 0:
            camera_config['dst_roi'] = [[int(pt[0]), int(pt[1])] for pt in self.bev_points]
        else:
            camera_config['dst_roi'] = []
        
        # 计算并保存H矩阵（确保是2D list格式）
        if len(self.camera_points) >= 4 and len(self.bev_points) >= 4:
            if len(self.camera_points) == len(self.bev_points):
                src_pts = np.array(self.camera_points, dtype=np.float32)
                dst_pts = np.array(self.bev_points, dtype=np.float32)
                
                # 使用findHomography代替getPerspectiveTransform，支持4个或更多点
                if len(self.camera_points) <= 7:
                    # 恰好4个点时，使用所有点直接计算
                    H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
                else:
                    # 超过4个点时，先尝试RANSAC算法
                    # 使用较大的阈值以包含更多内点，保持对应形状
                    H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10.0)
                    if mask is not None:
                        inliers = np.sum(mask)
                        total_points = len(self.camera_points)
                        inlier_ratio = inliers / total_points
                        print(f"RANSAC找到 {inliers}/{total_points} 个内点 ({inlier_ratio*100:.1f}%)")
                        
                        # 如果内点比例太低（少于80%），使用所有点直接计算以保持对应形状
                        if inlier_ratio < 0.8:
                            print(f"警告: 内点比例较低，可能破坏对应形状")
                            print(f"使用所有 {total_points} 个点直接计算H矩阵以保持完整对应关系...")
                            H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
                            print("已使用所有点计算H矩阵")
                
                if H is not None:
                    # 确保H是2D list格式
                    camera_config['H'] = H.tolist()
                    print(f"\nH矩阵计算成功:")
                    print(f"{H}")
                else:
                    print("错误: 无法计算H矩阵")
                    camera_config['H'] = []
            else:
                print("警告: 源点和目标点数量不匹配，无法计算H矩阵")
                camera_config['H'] = []
        else:
            # 如果点数不足，清空H矩阵
            camera_config['H'] = []
            if len(self.camera_points) > 0 or len(self.bev_points) > 0:
                print("警告: 至少需要4个点才能计算H矩阵，H矩阵已清空")
        
        # 确保src_roi, dst_roi, H都是标准的Python list格式（不是numpy数组或其他格式）
        # 递归转换嵌套结构为标准Python list
        def ensure_python_list(data):
            if isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, list):
                return [ensure_python_list(item) for item in data]
            elif isinstance(data, (tuple, np.generic)):
                return ensure_python_list(list(data))
            return data
        
        # 确保配置中的list格式正确
        camera_config['src_roi'] = ensure_python_list(camera_config.get('src_roi', []))
        camera_config['dst_roi'] = ensure_python_list(camera_config.get('dst_roi', []))
        camera_config['H'] = ensure_python_list(camera_config.get('H', []))
        
        # 创建自定义YAML dumper，为嵌套列表（2D/3D数组）使用flow_style（方括号格式）
        class FlowListDumper(yaml.SafeDumper):
            def represent_list(self, data):
                # 检查是否是嵌套列表（2D或3D数组）
                if data and isinstance(data[0], list):
                    # 对于嵌套列表，使用flow_style（方括号格式）: [[x, y], [x, y], ...]
                    return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
                else:
                    # 对于简单列表，使用block_style
                    return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
        
        # 注册自定义表示器
        FlowListDumper.add_representer(list, FlowListDumper.represent_list)
        
        # 保存到文件，嵌套列表会自动使用flow_style格式
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, Dumper=FlowListDumper, default_flow_style=False, 
                     allow_unicode=True, sort_keys=False)
        
        print(f"\n配置已保存到 {self.config_path}")
        print(f"摄像机: {self.position}")
        print(f"文件: {camera_config['file_name']}")
        print(f"源点数量: {len(self.camera_points)}")
        print(f"目标点数量: {len(self.bev_points)}")
        if camera_config['H']:
            print("H矩阵已计算并保存")
    
    def mouse_callback_camera(self, event, x, y, flags, param):
        """摄像头视图的鼠标回调函数"""
        # 记录当前活动窗口（包括鼠标移动和点击）
        if event in [cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
            self.active_window = 'camera'
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.closed_loop_camera:
                print("摄像头视图已闭合，按 'r' 重新开始或按 's' 保存")
                return
            
            self.camera_points.append((x, y))
            self.point_count = len(self.camera_points)
            print(f"摄像头视图 - 点 {self.point_count}: ({x}, {y})")
            
            # 重新绘制摄像头视图（从原始帧开始，确保不会重复叠加）
            self.redraw_camera_view()
            
            # 如果至少有3个点，提示可以close loop
            if len(self.camera_points) >= 3:
                print(f"提示: 已选择 {len(self.camera_points)} 个点，按 'c' 键闭合ROI（连接最后一个点和第一个点）")
    
    def mouse_callback_bev(self, event, x, y, flags, param):
        """BEV画布的鼠标回调函数"""
        # 记录当前活动窗口（包括鼠标移动和点击）
        if event in [cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
            self.active_window = 'bev'
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 在cal模式下，BEV ROI是固定的，不允许修改
            if self.cal_mode:
                print("校准模式：BEV ROI已固定，不能修改。请只调整摄像头ROI。")
                return
            
            if len(self.camera_points) == 0:
                print("请先在摄像头视图中选择ROI点")
                return
            
            if self.closed_loop_bev:
                print("BEV画布已闭合，按 'r' 重新开始或按 's' 保存")
                return
            
            if len(self.bev_points) < len(self.camera_points):
                self.bev_points.append((x, y))
                point_num = len(self.bev_points)
                print(f"BEV画布 - 点 {point_num}/{len(self.camera_points)}: ({x}, {y})")
                
                # 使用统一的绘制函数重新绘制BEV画布
                self.redraw_bev_view()
                
                # 如果所有点都已选择，提示可以close loop
                if len(self.bev_points) == len(self.camera_points):
                    print(f"\n✓ 所有 {len(self.bev_points)} 个点已选择完成！")
                    if len(self.bev_points) >= 3:
                        print("按 'c' 键闭合坐标组（连接最后一个点和第一个点），然后自动保存")
                    if len(self.bev_points) >= 4:
                        print("或按 's' 保存配置并计算H矩阵，按 'r' 重新开始，按 'q' 退出")
                    else:
                        print("警告: 至少需要4个点才能计算H矩阵，请继续选择点或按 'r' 重新开始")
            else:
                print("警告: 目标点数量已达到源点数量，请按 'c' 闭合或按 'r' 重新开始")
    
    def draw_roi_overlay(self):
        """在摄像头视图上绘制30%透明的绿色ROI区域"""
        if len(self.camera_points) < 3 or not self.closed_loop_camera:
            return
        
        # 创建覆盖层（从原始帧创建，确保透明度正确）
        overlay = self.original_frame.copy()
        
        # 将点转换为numpy数组格式（用于fillPoly）
        pts = np.array(self.camera_points, dtype=np.int32)
        
        # 填充ROI区域为绿色
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        
        # 使用addWeighted实现30%透明度叠加（alpha=0.3表示覆盖层30%不透明，即70%透明）
        # 将覆盖层叠加到当前显示图像上
        cv2.addWeighted(overlay, 0.3, self.camera_frame_display, 0.7, 0, self.camera_frame_display)
    
    def draw_bev_overlay(self):
        """在BEV画布上绘制30%透明的红色ROI区域"""
        if len(self.bev_points) < 3 or not self.closed_loop_bev:
            return
        
        # 创建覆盖层（从当前显示图像创建，这样可以在已有的点和线上叠加）
        overlay = self.bev_canvas_display.copy()
        
        # 将点转换为numpy数组格式（用于fillPoly）
        pts = np.array(self.bev_points, dtype=np.int32)
        
        # 填充ROI区域为红色（与BEV画布的点颜色一致）
        cv2.fillPoly(overlay, [pts], (255, 0, 0))
        
        # 使用addWeighted实现30%透明度叠加（alpha=0.3表示覆盖层30%不透明，即70%透明）
        cv2.addWeighted(overlay, 0.3, self.bev_canvas_display, 0.7, 0, self.bev_canvas_display)
    
    def draw_grid_lines(self, image, width, height, alpha=0.3):
        """在图像上绘制网格辅助线（横num_lines根纵num_lines根，半透明）"""
        # 创建覆盖层
        overlay = image.copy()
        
        # 计算线的间隔（num_lines+1个间隔，num_lines根线）
        num_intervals = self.num_lines + 1
        h_spacing = width / float(num_intervals)
        v_spacing = height / float(num_intervals)
        
        # 绘制横线（num_lines根）
        for i in range(1, self.num_lines + 1):
            y = int(i * v_spacing)
            cv2.line(overlay, (0, y), (width, y), (255, 255, 255), 1)
        
        # 绘制纵线（num_lines根）
        for i in range(1, self.num_lines + 1):
            x = int(i * h_spacing)
            cv2.line(overlay, (x, 0), (x, height), (255, 255, 255), 1)
        
        # 使用addWeighted实现半透明叠加
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    def redraw_camera_view(self):
        """重新绘制摄像头视图（包括点和线，如果闭合则包括半透明区域）"""
        # 从原始帧重新开始
        self.camera_frame_display = self.original_frame.copy()
        
        # 绘制所有点和线
        if self.camera_points:
            for i, pt in enumerate(self.camera_points):
                cv2.circle(self.camera_frame_display, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(self.camera_frame_display, 
                            self.camera_points[i-1], 
                            self.camera_points[i], 
                            (0, 255, 0), 2)
            
            # 如果已闭合，绘制闭合线和半透明区域
            if self.closed_loop_camera and len(self.camera_points) > 2:
                cv2.line(self.camera_frame_display, 
                        self.camera_points[-1], 
                        self.camera_points[0], 
                        (0, 255, 0), 2)
                # 绘制30%透明的绿色ROI区域
                self.draw_roi_overlay()
        
        # 最后绘制网格辅助线（确保在最上层显示）
        h, w = self.camera_frame_display.shape[:2]
        self.draw_grid_lines(self.camera_frame_display, w, h, alpha=0.3)
        
        cv2.imshow(self.camera_window, self.camera_frame_display)
    
    def close_loop_camera(self):
        """闭合摄像头视图的ROI（连接最后一个点和第一个点）"""
        if len(self.camera_points) < 3:
            print("警告: 至少需要3个点才能闭合ROI")
            return False
        
        if self.closed_loop_camera:
            print("摄像头视图已经闭合")
            return True
        
        # 先设置标志，然后重新绘制（这样redraw_camera_view可以检测到闭合状态）
        self.closed_loop_camera = True
        
        # 重新绘制摄像头视图（包括闭合线和半透明区域）
        if len(self.camera_points) > 2:
            self.redraw_camera_view()
            print(f"✓ 摄像头视图ROI已闭合（{len(self.camera_points)} 个点）")
            
            # 在cal模式下，如果BEV ROI已存在，自动重新计算H并更新BEV显示
            if self.cal_mode and self.closed_loop_bev and len(self.bev_points) >= 4:
                if len(self.camera_points) >= 4:
                    print("校准模式：自动重新计算H矩阵并更新BEV画面...")
                    self.update_bev_preview()
                    # 自动保存配置
                    self.save_config()
                    print("配置已自动保存")
                else:
                    print("警告: 至少需要4个点才能计算H矩阵")
            
            return True
        return False
    
    def draw_reference_rect(self):
        """在BEV画布上绘制参考矩形（长宽比1:2.54，即宽度:高度=1:2.54，位于中心，高度占画布的40%）"""
        # 计算矩形尺寸
        # 高度占画布的40%，宽度根据长宽比1:2.54计算（宽度:高度=1:2.54，所以宽度=高度/2.54）
        rect_height = int(self.bev_height * self.ref_width_ratio)
        rect_width = int(rect_height / self.ref_aspect_ratio)
        
        # 计算矩形中心位置（画布中心）
        center_x = self.bev_width // 2
        center_y = self.bev_height // 2
        
        # 计算矩形左上角坐标
        rect_x = center_x - rect_width // 2
        rect_y = center_y - rect_height // 2
        
        # 绘制矩形（白色边框）
        cv2.rectangle(self.bev_canvas_display, 
                     (rect_x, rect_y), 
                     (rect_x + rect_width, rect_y + rect_height), 
                     (255, 255, 255), 2)
    
    def draw_preview_projections(self):
        """在BEV画布上绘制所有已配置相机的ROI投射预览"""
        # 检查所有相机配置
        for cam_position in ['front', 'back', 'left', 'right']:
            cam_config = self.config['cameras'][cam_position]
            
            # 如果是当前摄像头且两个ROI都闭合，使用实时计算的H矩阵
            if (cam_position == self.position and 
                self.closed_loop_camera and self.closed_loop_bev and
                len(self.camera_points) >= 4 and len(self.bev_points) >= 4):
                # 使用当前摄像头和BEV的点实时计算H矩阵
                src_pts = np.array(self.camera_points, dtype=np.float32)
                dst_pts = np.array(self.bev_points, dtype=np.float32)
                
                if len(self.camera_points) == 4:
                    H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
                else:
                    H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                
                if H is None:
                    continue
                
                # 使用当前原始帧
                frame = self.original_frame
                dst_roi = np.array(self.bev_points, dtype=np.int32)
            else:
                # 检查是否有完整的配置（src_roi, dst_roi, H都存在且不为空）
                if not (cam_config.get('src_roi') and cam_config.get('dst_roi') and 
                    cam_config.get('H') and len(cam_config['src_roi']) >= 4 and 
                    len(cam_config['dst_roi']) >= 4 and len(cam_config['H']) == 3):
                    continue
                
                # 检查是否有对应的视频文件
                file_name = cam_config.get('file_name', '')
                if not file_name:
                    continue
                
                # 构建视频文件路径（与main函数中的路径构建方式一致）
                data_dir = Path(self.config_path).parent / 'data'
                video_path = data_dir / file_name
                
                if not video_path.exists():
                    continue
                
                try:
                    # 加载视频帧
                    frame = self.load_video_frame(str(video_path))
                    if frame is None or frame.size == 0:
                        continue
                except Exception as e:
                    print(f"警告: 无法加载 {cam_position} 相机的视频: {e}")
                    continue
                
                # 获取H矩阵
                H = np.array(cam_config['H'], dtype=np.float32)
                dst_roi = np.array(cam_config['dst_roi'], dtype=np.int32)
            
            try:
                # 使用H矩阵将整个图像投射到BEV画布
                warped_frame = cv2.warpPerspective(frame, H, (self.bev_width, self.bev_height))
                
                # 创建ROI掩码（在BEV画布上，使用dst_roi）
                mask = np.zeros((self.bev_height, self.bev_width), dtype=np.uint8)
                cv2.fillPoly(mask, [dst_roi], 255)
                
                # 将投射后的图像叠加到BEV画布上（只显示ROI区域内的部分）
                mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                self.bev_canvas_display = (self.bev_canvas_display * (1 - mask_3channel) + 
                                           warped_frame * mask_3channel).astype(np.uint8)
                
            except Exception as e:
                print(f"警告: 无法投射 {cam_position} 相机的ROI: {e}")
                continue
    
    def redraw_bev_view(self):
        """重新绘制BEV画布（包括点和线，如果闭合则包括半透明区域）"""
        # 从空白画布重新开始
        self.bev_canvas_display = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        
        # 绘制参考矩形
        self.draw_reference_rect()
        
        # 绘制预览投射（如果有完整配置的相机）
        self.draw_preview_projections()
        
        # 绘制提示文字
        cv2.putText(self.bev_canvas_display, 'BEV Canvas', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 根据状态显示不同的提示
        if len(self.camera_points) == 0:
            cv2.putText(self.bev_canvas_display, 'Click points for ROI', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif len(self.bev_points) < len(self.camera_points):
            remaining = len(self.camera_points) - len(self.bev_points)
            cv2.putText(self.bev_canvas_display, f'Click {remaining} more point(s)', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif len(self.bev_points) == len(self.camera_points) and not self.closed_loop_bev:
            cv2.putText(self.bev_canvas_display, 'Press \'c\' to close loop', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif self.closed_loop_bev:
            cv2.putText(self.bev_canvas_display, 'Closed loop - Press \'s\' to save', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制所有点和线
        if self.bev_points:
            for i, pt in enumerate(self.bev_points):
                cv2.circle(self.bev_canvas_display, pt, 5, (255, 0, 0), -1)
                if i > 0:
                    cv2.line(self.bev_canvas_display, 
                            self.bev_points[i-1], 
                            self.bev_points[i], 
                            (255, 0, 0), 2)
            
            # 如果已闭合，绘制闭合线和半透明区域
            if self.closed_loop_bev and len(self.bev_points) > 2:
                # 绘制闭合线（连接最后一个点和第一个点）
                cv2.line(self.bev_canvas_display, 
                        self.bev_points[-1], 
                        self.bev_points[0], 
                        (255, 0, 0), 2)
                # 绘制30%透明的红色ROI区域
                self.draw_bev_overlay()
        
        # 最后绘制网格辅助线（确保在最上层显示）
        self.draw_grid_lines(self.bev_canvas_display, self.bev_width, self.bev_height, alpha=0.3)
        
        cv2.imshow(self.bev_window, self.bev_canvas_display)
    
    def close_loop_bev(self):
        """闭合BEV画布的坐标组（连接最后一个点和第一个点）"""
        if len(self.bev_points) < 3:
            print("警告: 至少需要3个点才能闭合坐标组")
            return False
        
        if len(self.bev_points) != len(self.camera_points):
            print("警告: BEV画布的点数必须与摄像头视图的点数相同")
            return False
        
        if self.closed_loop_bev:
            print("BEV画布已经闭合")
            return True
        
        # 重新绘制BEV画布（包括闭合线和半透明区域）
        if len(self.bev_points) > 2:
            self.redraw_bev_view()
            self.closed_loop_bev = True
            print(f"✓ BEV画布坐标组已闭合（{len(self.bev_points)} 个点）")
            
            # 如果两个都闭合了，重新计算H并更新BEV显示
            if self.closed_loop_camera and self.closed_loop_bev:
                if len(self.camera_points) >= 4:
                    print("\n两个坐标组都已闭合，重新计算H矩阵并更新BEV显示...")
                    # 重新计算H矩阵
                    src_pts = np.array(self.camera_points, dtype=np.float32)
                    dst_pts = np.array(self.bev_points, dtype=np.float32)
                    
                    if len(self.camera_points) == 4:
                        H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
                    else:
                        # 先尝试RANSAC算法，使用较大的阈值以包含更多内点
                        H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10.0)
                        if mask is not None:
                            inliers = np.sum(mask)
                            total_points = len(self.camera_points)
                            inlier_ratio = inliers / total_points
                            print(f"RANSAC找到 {inliers}/{total_points} 个内点 ({inlier_ratio*100:.1f}%)")
                            
                            # 如果内点比例太低（少于80%），使用所有点直接计算以保持对应形状
                            if inlier_ratio < 0.8:
                                print(f"警告: 内点比例较低，可能破坏对应形状")
                                print(f"使用所有 {total_points} 个点直接计算H矩阵以保持完整对应关系...")
                                H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
                                print("已使用所有点计算H矩阵")
                    
                    if H is not None:
                        print(f"H矩阵计算成功:")
                        print(f"{H}")
                        # 更新BEV显示（使用新的H矩阵重新绘制）
                        self.redraw_bev_view()
                        print("BEV画面已更新（使用新的H矩阵）")
                    else:
                        print("错误: 无法计算H矩阵")
                    
                    # 自动保存配置
                    print("自动保存配置...")
                    self.save_config()
                    print("配置已自动保存！")
                else:
                    print("警告: 至少需要4个点才能计算H矩阵，配置已保存但H矩阵未计算")
                    self.save_config()
            return True
        return False
    
    def load_video_frame(self, video_path, frame_number=0):
        """从视频中加载指定帧"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 设置到指定帧
        if frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # 读取帧
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"无法从视频中读取帧: {video_path} (帧号: {frame_number})")
        
        return frame
    
    def update_bev_preview(self):
        """更新BEV画布的预览（在cal模式下，当摄像头ROI闭合后自动更新）"""
        if not self.cal_mode:
            return
        
        if not self.closed_loop_camera or not self.closed_loop_bev:
            return
        
        if len(self.camera_points) < 4 or len(self.bev_points) < 4:
            return
        
        # 重新计算H矩阵
        src_pts = np.array(self.camera_points, dtype=np.float32)
        dst_pts = np.array(self.bev_points, dtype=np.float32)
        
        if len(self.camera_points) == 4:
            H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
        else:
            H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        
        if H is not None:
            # 重新绘制BEV画布（这会自动使用新的H矩阵更新预览）
            self.redraw_bev_view()
            print("BEV画面已更新（使用新的H矩阵）")
    
    def run(self):
        """运行工具主循环"""
        # 加载视频帧
        print(f"加载视频: {self.video_path}")
        if self.start_frame > 0:
            print(f"从第 {self.start_frame} 帧开始显示")
        frame = self.load_video_frame(self.video_path, self.start_frame)
        
        # 检查是否有已保存的配置
        camera_config = self.config['cameras'][self.position]
        if (camera_config.get('file_name') == f'sample{self.sample}_{self.position}.mp4' and 
            camera_config.get('src_roi') and camera_config.get('dst_roi')):
            print(f"\n发现已保存的配置:")
            print(f"  源点: {len(camera_config['src_roi'])} 个")
            print(f"  目标点: {len(camera_config['dst_roi'])} 个")
            
            if self.cal_mode:
                print("校准模式：将加载BEV ROI，摄像头ROI需要重新选择")
                # 在cal模式下，只加载BEV ROI
                self.bev_points = [tuple(pt) for pt in camera_config['dst_roi']]
                if camera_config.get('H'):
                    self.closed_loop_bev = True
                print("已加载BEV ROI配置")
            else:
                response = input("是否加载已有配置？(y/n): ").strip().lower()
                if response == 'y':
                    self.camera_points = [tuple(pt) for pt in camera_config['src_roi']]
                    self.bev_points = [tuple(pt) for pt in camera_config['dst_roi']]
                    print("已加载已有配置")
        
        # 保存原始帧用于重新绘制
        self.original_frame = frame.copy()
        
        # 创建显示用的图像副本
        self.camera_frame_display = frame.copy()
        self.bev_canvas_display = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        
        # 如果已加载配置，检查是否已闭合
        if self.camera_points and camera_config.get('H'):
            self.closed_loop_camera = True
        if self.bev_points and camera_config.get('H'):
            self.closed_loop_bev = True
        
        # 使用统一的绘制函数绘制摄像头视图和BEV画布
        self.redraw_camera_view()
        self.redraw_bev_view()
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.camera_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.bev_window, cv2.WINDOW_NORMAL)
        
        cv2.setMouseCallback(self.camera_window, self.mouse_callback_camera)
        cv2.setMouseCallback(self.bev_window, self.mouse_callback_bev)
        
        # 显示初始图像
        cv2.imshow(self.camera_window, self.camera_frame_display)
        cv2.imshow(self.bev_window, self.bev_canvas_display)
        
        print("\n" + "="*60)
        print("BEV Homography Matrix Tool")
        print("="*60)
        if self.cal_mode:
            print("【校准模式】BEV ROI已固定，只调整摄像头ROI")
        print("\n使用说明:")
        if self.cal_mode:
            print("1. 在摄像头视图中点击鼠标选择ROI点（至少3个点，建议按顺序点击）")
            print("2. 选择完最后一个点后，按 'c' 键闭合摄像头ROI")
            print("3. 闭合后会自动重新计算H矩阵并更新BEV画面（如果BEV ROI已存在）")
            print("4. 按 'r' 重新选择摄像头ROI（BEV ROI保持不变）")
            print("5. 按 'k' 重置当前焦点窗口的ROI（点击窗口后按k）")
            print("6. 按 'l' 撤销最后一个选择的点")
            print("7. 按 's' 手动保存配置")
        else:
            print("1. 在摄像头视图中点击鼠标选择ROI点（至少3个点，建议按顺序点击）")
            print("2. 在BEV画布上点击对应数量的点（按相同顺序）")
            print("3. 选择完最后一个点后，按 'c' 键闭合坐标组（连接最后一个点和第一个点）")
            print("4. 闭合后会自动保存配置并计算H矩阵（如果点数>=4）")
            print("5. 或按 's' 手动保存配置并计算H矩阵")
            print("6. 按 'r' 重新开始选择点（重置所有）")
            print("7. 按 'k' 重置当前焦点窗口的ROI（点击窗口后按k，只重置该窗口）")
            print("8. 按 'l' 撤销最后一个选择的点")
        print("9. 按 'q' 退出")
        print("\n提示: 摄像头视图中的点用绿色标记，BEV画布中的点用红色标记")
        print("提示: 点击窗口后按 'k' 可以重置该窗口的ROI，不影响另一个窗口")
        print("="*60)
        if self.cal_mode:
            print("\n开始选择摄像头视图的ROI点（BEV ROI已固定）...")
        else:
            print("\n开始选择摄像头视图的ROI点...")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # 更新BEV画布（使用统一的绘制函数）
            if len(self.camera_points) > 0:
                self.redraw_bev_view()
            
            if key == ord('q'):
                print("退出工具")
                break
            elif key == ord('c'):
                # Close loop功能
                # 优先闭合摄像头视图
                if len(self.camera_points) >= 3 and not self.closed_loop_camera:
                    self.close_loop_camera()
                    if len(self.camera_points) >= 3:
                        print("提示: 现在可以在BEV画布上选择对应数量的点，然后再次按 'c' 闭合BEV画布")
                # 如果摄像头视图已闭合，且BEV画布点数匹配，则闭合BEV画布
                elif self.closed_loop_camera and len(self.bev_points) >= 3 and len(self.bev_points) == len(self.camera_points) and not self.closed_loop_bev:
                    self.close_loop_bev()
                elif self.closed_loop_camera and self.closed_loop_bev:
                    print("两个坐标组都已闭合，配置已保存")
                else:
                    if len(self.camera_points) < 3:
                        print("错误: 摄像头视图至少需要3个点才能闭合")
                    elif not self.closed_loop_camera:
                        print("错误: 请先闭合摄像头视图的ROI")
                    elif len(self.bev_points) < 3:
                        print("错误: BEV画布至少需要3个点才能闭合")
                    elif len(self.bev_points) != len(self.camera_points):
                        print(f"错误: BEV画布的点数({len(self.bev_points)})必须与摄像头视图的点数({len(self.camera_points)})相同")
            elif key == ord('s'):
                if len(self.camera_points) >= 4 and len(self.bev_points) >= 4:
                    if len(self.camera_points) == len(self.bev_points):
                        self.save_config()
                        print("配置已保存！")
                    else:
                        print("错误: 源点和目标点数量必须相同")
                else:
                    print("错误: 至少需要4个点才能计算H矩阵")
            elif key == ord('l'):
                # 撤销最后一个选择的点
                if self.closed_loop_camera or self.closed_loop_bev:
                    print("已闭合的ROI不能撤销点，请先按 'r' 重新开始")
                elif len(self.bev_points) > 0 and len(self.bev_points) == len(self.camera_points):
                    # 如果BEV点数和摄像头点数相同，撤销BEV的最后一个点
                    removed_point = self.bev_points.pop()
                    print(f"已撤销BEV画布的最后一个点: {removed_point}")
                    self.redraw_bev_view()
                elif len(self.camera_points) > 0:
                    # 撤销摄像头的最后一个点
                    removed_point = self.camera_points.pop()
                    self.point_count = len(self.camera_points)
                    print(f"已撤销摄像头视图的最后一个点: {removed_point}")
                    # 如果BEV点数超过摄像头点数，也需要撤销BEV的点
                    if len(self.bev_points) > len(self.camera_points):
                        self.bev_points.pop()
                    self.redraw_camera_view()
                    self.redraw_bev_view()
                else:
                    print("没有可撤销的点")
            elif key == ord('k'):
                # 重置当前焦点窗口的ROI
                if self.active_window == 'camera':
                    # 重置摄像头ROI
                    if len(self.camera_points) > 0 or self.closed_loop_camera:
                        print("重置摄像头ROI...")
                        self.camera_points = []
                        self.point_count = 0
                        self.closed_loop_camera = False
                        # 如果BEV点数超过摄像头点数，也需要重置BEV的点
                        if len(self.bev_points) > len(self.camera_points):
                            self.bev_points = []
                            self.closed_loop_bev = False
                        self.redraw_camera_view()
                        if len(self.bev_points) == 0:
                            self.redraw_bev_view()
                        print("摄像头ROI已重置")
                    else:
                        print("摄像头ROI已经是空的")
                elif self.active_window == 'bev':
                    # 重置BEV ROI
                    if self.cal_mode:
                        print("校准模式：BEV ROI已固定，不能重置")
                    elif len(self.bev_points) > 0 or self.closed_loop_bev:
                        print("重置BEV ROI...")
                        self.bev_points = []
                        self.closed_loop_bev = False
                        self.redraw_bev_view()
                        print("BEV ROI已重置")
                    else:
                        print("BEV ROI已经是空的")
                else:
                    print("请先点击摄像头窗口或BEV窗口以激活，然后按 'k' 重置对应窗口的ROI")
            elif key == ord('r'):
                if self.cal_mode:
                    # 校准模式：只重置摄像头ROI，保持BEV ROI不变
                    print("校准模式：重新选择摄像头ROI（BEV ROI保持不变）...")
                    self.camera_points = []
                    self.point_count = 0
                    self.closed_loop_camera = False
                    # 只重新绘制摄像头视图
                    self.redraw_camera_view()
                else:
                    # 普通模式：重置所有点
                    print("重新开始选择点...")
                    self.camera_points = []
                    self.bev_points = []
                    self.point_count = 0
                    self.closed_loop_camera = False
                    self.closed_loop_bev = False
                    # 重新绘制摄像头视图和BEV画布（从原始帧/空白画布开始）
                    self.redraw_camera_view()
                    self.redraw_bev_view()
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='BEV Homography Matrix Tool')
    parser.add_argument('--sample', type=int, required=True, 
                       help='Sample number (e.g., 1 or 2)')
    parser.add_argument('--position', type=str, required=True, 
                       choices=['front', 'back', 'left', 'right'],
                       help='Camera position: front, back, left, or right')
    parser.add_argument('--lines', type=int, default=6,
                       help='Number of grid lines in each direction (default: 6)')
    parser.add_argument('--cal', type=str, default='False',
                       choices=['True', 'False', 'true', 'false'],
                       help='Calibration mode: keep BEV ROI fixed, only adjust camera ROI (default: False)')
    parser.add_argument('--startfrom', type=int, default=0,
                       help='Start frame number (default: 0)')
    
    args = parser.parse_args()
    
    # 构建文件路径
    data_dir = Path(__file__).parent / 'data'
    video_filename = f'sample{args.sample}_{args.position}.mp4'
    video_path = data_dir / video_filename
    
    config_path = Path(__file__).parent / 'config.yaml'
    
    # 检查视频文件是否存在
    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    # 解析cal参数
    cal_mode = args.cal.lower() == 'true'
    
    # 创建并运行工具
    tool = BEVHomographyTool(str(video_path), str(config_path), args.position, 
                            args.sample, args.lines, cal_mode, args.startfrom)
    tool.run()


if __name__ == '__main__':
    main()
