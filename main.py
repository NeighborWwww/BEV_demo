#!/usr/bin/env python3
"""
BEV Video Composition and Playback Tool
用于合成和播放BEV（Bird Eye View）视频
"""

import cv2
import numpy as np
import yaml
import argparse
import os
import time
from pathlib import Path
from datetime import datetime
from collections import deque


class BEVVideoPlayer:
    def __init__(self, config_path, sample, offsets=None, save_video=False):
        self.config_path = config_path
        self.sample = sample
        self.save_video = save_video
        
        # 加载配置
        self.config = self.load_config()
        
        # 获取BEV画布尺寸
        self.bev_width = self.config['bev']['canvas_width']
        self.bev_height = self.config['bev']['canvas_height']
        
        # 设置偏移量（默认0）
        self.offsets = offsets or {'front': 0, 'back': 0, 'left': 0, 'right': 0}
        
        # 摄像头位置
        self.positions = ['front', 'back', 'left', 'right']
        
        # 视频捕获对象
        self.caps = {}
        self.current_frames = {}
        self.homography_matrices = {}
        self.current_frame_numbers = {}  # 记录每个视频的当前帧号
        self.initial_offsets = {}  # 记录每个视频的初始offset
        
        # 初始化视频捕获
        self.init_video_captures()
        
        # 视频写入器（如果需要保存）
        self.video_writer = None
        self.loop_completed = False  # 标记是否完成了一个循环
        self.is_first_loop = True  # 标记是否是第一个循环
        self.output_path = None
        self.video_width = None
        self.video_height = None
        # 延迟初始化视频写入器（在第一次合成帧后）
        
        # 性能统计相关
        self.enable_perf_stats = False  # 是否启用性能统计
        self.perf_stats = {
            'frame_count': 0,
            'total_time': 0.0,
            'read_frames_time': 0.0,
            'compose_bev_time': 0.0,
            'compose_grid_time': 0.0,
            'compose_final_time': 0.0,
            'display_time': 0.0,
            'write_time': 0.0,
            'fps_history': deque(maxlen=30),  # 保存最近30帧的FPS
        }
    
    def load_config(self):
        """加载YAML配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
    
    def init_video_captures(self):
        """初始化所有视频捕获对象"""
        data_dir = Path(self.config_path).parent / 'data'
        
        for position in self.positions:
            cam_config = self.config['cameras'][position]
            
            # 优先使用config中的文件名，如果没有则根据sample参数构建
            file_name = cam_config.get('file_name', '')
            if not file_name:
                # 根据sample参数动态构建文件名
                file_name = f'sample{self.sample}_{position}.mp4'
            
            video_path = data_dir / file_name
            
            if not video_path.exists():
                print(f"警告: 视频文件不存在: {video_path}，跳过")
                continue
            
            # 打开视频
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"警告: 无法打开视频: {video_path}，跳过")
                continue
            
            # 设置到偏移帧（只使用命令行参数，与config无关）
            offset = self.offsets.get(position, 0)
            
            if offset > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
            
            # 记录初始offset和当前帧号
            self.initial_offsets[position] = offset
            self.current_frame_numbers[position] = offset
            
            self.caps[position] = cap
            
            # 加载H矩阵
            H = cam_config.get('H', [])
            if H and len(H) == 3 and len(H[0]) == 3:
                self.homography_matrices[position] = np.array(H, dtype=np.float32)
            else:
                print(f"警告: {position} 相机没有有效的H矩阵，跳过BEV投射")
    
    def init_video_writer(self, frame):
        """初始化视频写入器（使用第一帧来确定尺寸）"""
        if self.video_writer is not None:
            return  # 已经初始化过了
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"result_{timestamp}.mp4"
        output_path = Path(self.config_path).parent / output_filename
        
        if frame is None or frame.size == 0:
            print("错误: 无法获取帧，视频保存初始化失败")
            return
        
        actual_height, actual_width = frame.shape[:2]
        
        # 获取FPS（使用第一个视频的FPS）
        first_cap = next(iter(self.caps.values()))
        fps = int(first_cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # 默认FPS
        
        # 尝试使用不同的编码器（mp4v可能在某些系统上不可用）
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'XVID'),
            cv2.VideoWriter_fourcc(*'avc1'),
            cv2.VideoWriter_fourcc(*'H264'),
        ]
        
        self.video_writer = None
        for fourcc in fourcc_options:
            self.video_writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (actual_width, actual_height)
            )
            if self.video_writer.isOpened():
                print(f"视频写入器初始化成功，使用编码器: {fourcc}")
                break
            else:
                if self.video_writer:
                    self.video_writer.release()
                self.video_writer = None
        
        if self.video_writer is None or not self.video_writer.isOpened():
            print(f"错误: 无法初始化视频写入器，视频将不会保存")
            print(f"尝试保存到: {output_path}")
            print(f"视频尺寸: {actual_width}x{actual_height}, FPS: {fps}")
            self.video_writer = None
        else:
            self.output_path = output_path
            self.video_width = actual_width
            self.video_height = actual_height
            print(f"视频将保存到: {output_path}")
            print(f"视频尺寸: {actual_width}x{actual_height}, FPS: {fps}")
    
    def read_frames(self):
        """读取所有视频的当前帧"""
        start_time = time.time() if self.enable_perf_stats else None
        
        all_ended = True
        ended_positions = []
        
        # 先读取所有视频的帧
        for position, cap in self.caps.items():
            ret, frame = cap.read()
            if ret:
                self.current_frames[position] = frame
                # 更新当前帧号
                self.current_frame_numbers[position] = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                all_ended = False
            else:
                # 视频结束，记录结束的位置
                ended_positions.append(position)
        
        # 如果有视频结束，检查是否所有视频都结束了
        if ended_positions:
            # 如果所有视频都结束了，统一重置所有视频到初始状态
            if all_ended:
                # 如果正在保存视频且是第一个循环，标记循环完成
                if self.save_video and self.is_first_loop:
                    self.loop_completed = True
                    self.is_first_loop = False
                    print("第一个循环完成，视频保存将停止...")
                
                print("所有视频播放完毕，重置到初始状态（从各自设定的offset开始）...")
                for position in self.positions:
                    if position in self.caps:
                        # 重置到初始offset
                        initial_offset = self.initial_offsets.get(position, 0)
                        cap = self.caps[position]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, initial_offset)
                        ret, frame = cap.read()
                        if ret:
                            self.current_frames[position] = frame
                            self.current_frame_numbers[position] = initial_offset
                            all_ended = False
            else:
                # 部分视频结束，等待所有视频都结束后再统一重置
                # 对于已结束的视频，保持在最后一帧（不读取新帧）
                # 继续读取未结束的视频，直到所有视频都结束
                for position in ended_positions:
                    # 保持最后一帧（如果存在）
                    if position not in self.current_frames:
                        # 如果当前没有帧，尝试读取最后一帧
                        cap = self.caps[position]
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if total_frames > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                            ret, frame = cap.read()
                            if ret:
                                self.current_frames[position] = frame
                                self.current_frame_numbers[position] = total_frames - 1
        
        if self.enable_perf_stats and start_time:
            self.perf_stats['read_frames_time'] += time.time() - start_time
        
        return not all_ended  # 返回是否还有视频在播放
    
    def compose_bev_canvas(self):
        """合成BEV画布"""
        start_time = time.time() if self.enable_perf_stats else None
        
        # 创建BEV画布
        bev_canvas = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        
        # 对每个摄像头进行透视变换并叠加
        for position in self.positions:
            if position not in self.current_frames:
                continue
            
            if position not in self.homography_matrices:
                continue
            
            frame = self.current_frames[position]
            H = self.homography_matrices[position]
            cam_config = self.config['cameras'][position]
            
            # 使用H矩阵将图像投射到BEV画布
            warped_frame = cv2.warpPerspective(frame, H, (self.bev_width, self.bev_height))
            
            # 获取dst_roi作为掩码
            dst_roi = cam_config.get('dst_roi', [])
            if dst_roi and len(dst_roi) >= 4:
                # 创建ROI掩码
                mask = np.zeros((self.bev_height, self.bev_width), dtype=np.uint8)
                dst_roi_array = np.array(dst_roi, dtype=np.int32)
                cv2.fillPoly(mask, [dst_roi_array], 255)
                
                # 将投射后的图像叠加到BEV画布上（只显示ROI区域内的部分）
                mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                bev_canvas = (bev_canvas * (1 - mask_3channel) + 
                             warped_frame * mask_3channel).astype(np.uint8)
        
        if self.enable_perf_stats and start_time:
            self.perf_stats['compose_bev_time'] += time.time() - start_time
        
        return bev_canvas
    
    def compose_grid_view(self):
        """合成四宫格视图（以BEV高度为基准调整窗口大小）"""
        start_time = time.time() if self.enable_perf_stats else None
        
        # 计算每个摄像头窗口的目标尺寸（以BEV高度为基准）
        target_cam_height = self.bev_height // 2
        
        frames = []
        labels = ['Front', 'Back', 'Left', 'Right']
        
        for i, position in enumerate(self.positions):
            if position in self.current_frames:
                frame = self.current_frames[position].copy()
                
                # 调整帧大小以匹配目标高度（保持宽高比）
                original_height, original_width = frame.shape[:2]
                scale = target_cam_height / original_height
                target_cam_width = int(original_width * scale)
                frame = cv2.resize(frame, (target_cam_width, target_cam_height))
                
                # 添加标签和帧号信息
                label_text = labels[i]
                frame_num = self.current_frame_numbers.get(position, 0)
                offset = self.offsets.get(position, 0)
                
                # 构建显示文本
                info_text = f"{label_text} Frame:{frame_num}"
                if offset > 0:
                    info_text += f" offset={offset}"
                
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                frames.append(frame)
            else:
                # 如果某个视频不存在，创建黑色占位符
                # 使用第一个有效帧的尺寸，如果没有则使用默认尺寸
                if frames:
                    target_cam_width = frames[0].shape[1]
                else:
                    # 如果没有有效帧，使用第一个视频的原始尺寸计算
                    if self.caps:
                        first_cap = next(iter(self.caps.values()))
                        original_cam_height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        original_cam_width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        scale = target_cam_height / original_cam_height
                        target_cam_width = int(original_cam_width * scale)
                    else:
                        target_cam_width = 640
                
                placeholder = np.zeros((target_cam_height, target_cam_width, 3), dtype=np.uint8)
                
                # 添加标签
                label_text = labels[i]
                offset = self.offsets.get(position, 0)
                info_text = f"{label_text} (No video)"
                if offset > 0:
                    info_text += f" offset={offset}"
                
                cv2.putText(placeholder, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                frames.append(placeholder)
        
        # 确保所有帧宽度相同（取最大宽度）
        if frames:
            max_width = max(frame.shape[1] for frame in frames)
            resized_frames = []
            for frame in frames:
                if frame.shape[1] != max_width:
                    # 保持高度，调整宽度（居中裁剪或填充）
                    h, w = frame.shape[:2]
                    if w < max_width:
                        # 填充黑色
                        pad_left = (max_width - w) // 2
                        pad_right = max_width - w - pad_left
                        frame = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right, 
                                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    else:
                        # 裁剪
                        crop_left = (w - max_width) // 2
                        frame = frame[:, crop_left:crop_left + max_width]
                resized_frames.append(frame)
            
            # 创建2x2网格
            top_row = np.hstack([resized_frames[0], resized_frames[1]])  # front, back
            bottom_row = np.hstack([resized_frames[2], resized_frames[3]])  # left, right
            grid = np.vstack([top_row, bottom_row])
            
            if self.enable_perf_stats and start_time:
                self.perf_stats['compose_grid_time'] += time.time() - start_time
            
            return grid
        else:
            if self.enable_perf_stats and start_time:
                self.perf_stats['compose_grid_time'] += time.time() - start_time
            return np.zeros((self.bev_height, 640, 3), dtype=np.uint8)
    
    def compose_final_frame(self):
        """合成最终显示帧（四宫格 + BEV画布）"""
        grid_view = self.compose_grid_view()
        bev_canvas = self.compose_bev_canvas()
        
        # 开始计时合成最终帧本身的操作（不包括子函数）
        start_time = time.time() if self.enable_perf_stats else None
        
        # 确保BEV画布高度与四宫格高度一致（应该已经是，但为了安全起见）
        grid_height, grid_width = grid_view.shape[:2]
        if bev_canvas.shape[0] != grid_height:
            bev_canvas = cv2.resize(bev_canvas, (self.bev_width, grid_height))
        
        # 在BEV画布上添加标签
        cv2.putText(bev_canvas, 'BEV View', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 水平拼接
        final_frame = np.hstack([grid_view, bev_canvas])
        
        if self.enable_perf_stats and start_time:
            self.perf_stats['compose_final_time'] += time.time() - start_time
        
        return final_frame
    
    def display_frame_info(self, frame):
        """在画面上显示当前帧号信息（暂停时显示）"""
        # 在画面右上角显示帧号信息
        info_lines = []
        info_lines.append("Frame Info:")
        for position in self.positions:
            frame_num = self.current_frame_numbers.get(position, 0)
            offset = self.offsets.get(position, 0)
            label = position.capitalize()
            info_text = f"{label}: {frame_num}"
            if offset > 0:
                info_text += f" (offset={offset})"
            info_lines.append(info_text)
        
        # 计算文本位置
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (frame.shape[1] - 300, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def display_perf_stats(self, frame):
        """在画面上显示性能统计信息"""
        if not self.enable_perf_stats:
            return
        
        stats = self.perf_stats
        frame_count = stats['frame_count']
        
        if frame_count == 0:
            return
        
        # 计算平均时间
        avg_read = stats['read_frames_time'] / frame_count * 1000  # ms
        avg_bev = stats['compose_bev_time'] / frame_count * 1000
        avg_grid = stats['compose_grid_time'] / frame_count * 1000
        avg_final = stats['compose_final_time'] / frame_count * 1000
        avg_total = stats['total_time'] / frame_count * 1000
        
        # 计算当前FPS（最近30帧的平均）
        if len(stats['fps_history']) > 0:
            current_fps = sum(stats['fps_history']) / len(stats['fps_history'])
        else:
            current_fps = 0
        
        # 在画面左上角显示性能信息
        perf_lines = [
            "=== Performance Stats ===",
            f"FPS: {current_fps:.1f}",
            f"Frame: {frame_count}",
            "",
            "Avg Time (ms):",
            f"  Read: {avg_read:.2f}",
            f"  BEV: {avg_bev:.2f}",
            f"  Grid: {avg_grid:.2f}",
            f"  Final: {avg_final:.2f}",
            f"  Total: {avg_total:.2f}",
        ]
        
        y_offset = 30
        for i, line in enumerate(perf_lines):
            color = (0, 255, 0) if i == 1 else (255, 255, 255)  # FPS用绿色
            cv2.putText(frame, line, (10, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def print_perf_summary(self):
        """打印性能统计摘要"""
        if not self.enable_perf_stats:
            return
        
        stats = self.perf_stats
        frame_count = stats['frame_count']
        
        if frame_count == 0:
            print("没有性能统计数据")
            return
        
        print("\n" + "="*50)
        print("性能统计摘要")
        print("="*50)
        print(f"总帧数: {frame_count}")
        print(f"总耗时: {stats['total_time']:.2f} 秒")
        print(f"平均FPS: {frame_count / stats['total_time']:.2f}")
        
        print("\n各阶段平均耗时 (ms):")
        print(f"  读取帧: {stats['read_frames_time'] / frame_count * 1000:.2f}")
        print(f"  合成BEV: {stats['compose_bev_time'] / frame_count * 1000:.2f}")
        print(f"  合成网格: {stats['compose_grid_time'] / frame_count * 1000:.2f}")
        print(f"  合成最终帧: {stats['compose_final_time'] / frame_count * 1000:.2f}")
        print(f"  总耗时: {stats['total_time'] / frame_count * 1000:.2f}")
        
        # 计算各阶段占比
        total_processing = (stats['read_frames_time'] + stats['compose_bev_time'] + 
                          stats['compose_grid_time'] + stats['compose_final_time'])
        if total_processing > 0:
            print("\n各阶段耗时占比 (%):")
            print(f"  读取帧: {stats['read_frames_time'] / total_processing * 100:.1f}%")
            print(f"  合成BEV: {stats['compose_bev_time'] / total_processing * 100:.1f}%")
            print(f"  合成网格: {stats['compose_grid_time'] / total_processing * 100:.1f}%")
            print(f"  合成最终帧: {stats['compose_final_time'] / total_processing * 100:.1f}%")
        
        print("="*50)
    
    def run(self):
        """运行视频播放"""
        print("开始播放视频...")
        if self.save_video:
            print("视频保存已启用，将保存一个完整循环的视频")
        if self.enable_perf_stats:
            print("性能统计已启用，将在画面上显示性能信息")
        print("按 'q' 退出，按 'p' 暂停/继续（暂停时可记录帧号用于调整标定）")
        
        paused = False
        frame_start_time = None
        
        while True:
            # 如果保存视频且已完成一个循环，停止保存并退出
            if self.save_video and self.loop_completed:
                print("一个循环的视频已保存完成")
                break
            
            # 开始计时（用于性能统计）
            if self.enable_perf_stats:
                frame_start_time = time.time()
            
            if not paused:
                # 读取所有帧
                has_frames = self.read_frames()
                if not has_frames:
                    # 如果没有保存视频，继续循环播放
                    if not self.save_video:
                        continue
                    else:
                        # 如果正在保存视频，等待循环完成
                        if not self.loop_completed:
                            continue
                        else:
                            break
            
            # 合成最终帧
            compose_start = time.time() if self.enable_perf_stats else None
            final_frame = self.compose_final_frame()
            
            # 如果保存视频且视频写入器未初始化，现在初始化
            if self.save_video and self.video_writer is None and final_frame is not None:
                self.init_video_writer(final_frame)
            
            # 如果暂停，显示帧号信息
            if paused:
                self.display_frame_info(final_frame)
                # 在画面中央显示暂停提示
                h, w = final_frame.shape[:2]
                pause_text = "PAUSED - Press 'p' to resume"
                text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h // 2
                cv2.putText(final_frame, pause_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示性能统计信息
            if self.enable_perf_stats:
                self.display_perf_stats(final_frame)
            
            # 显示
            display_start = time.time() if self.enable_perf_stats else None
            cv2.imshow('BEV Video Player', final_frame)
            if self.enable_perf_stats and display_start:
                self.perf_stats['display_time'] += time.time() - display_start
            
            # 保存视频（如果需要，暂停时不保存，且未完成循环）
            if self.video_writer and not paused and not self.loop_completed:
                write_start = time.time() if self.enable_perf_stats else None
                # 确保帧的尺寸与VideoWriter期望的尺寸一致
                if final_frame.shape[:2] != (self.video_height, self.video_width):
                    final_frame = cv2.resize(final_frame, (self.video_width, self.video_height))
                
                # 确保帧是uint8类型
                if final_frame.dtype != np.uint8:
                    final_frame = final_frame.astype(np.uint8)
                
                # 写入帧
                success = self.video_writer.write(final_frame)
                if not success:
                    print(f"警告: 写入帧失败")
                if self.enable_perf_stats and write_start:
                    self.perf_stats['write_time'] += time.time() - write_start
            
            # 更新性能统计
            if self.enable_perf_stats and frame_start_time and not paused:
                frame_time = time.time() - frame_start_time
                self.perf_stats['total_time'] += frame_time
                self.perf_stats['frame_count'] += 1
                
                # 计算FPS并添加到历史记录
                if frame_time > 0:
                    fps = 1.0 / frame_time
                    self.perf_stats['fps_history'].append(fps)
            
            # 处理按键
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                if paused:
                    print("\n=== 暂停 ===")
                    print("当前帧号信息（可用于调整标定参数）：")
                    for position in self.positions:
                        frame_num = self.current_frame_numbers.get(position, 0)
                        offset = self.offsets.get(position, 0)
                        print(f"  {position.capitalize()}: Frame {frame_num}" + 
                              (f" (offset={offset})" if offset > 0 else ""))
                    print("按 'p' 继续播放")
                else:
                    print("继续播放...")
        
        # 清理资源
        for cap in self.caps.values():
            cap.release()
        
        if self.video_writer:
            self.video_writer.release()
            if hasattr(self, 'output_path'):
                if self.output_path.exists():
                    file_size = self.output_path.stat().st_size
                    print(f"视频文件已保存: {self.output_path}")
                    print(f"文件大小: {file_size / (1024*1024):.2f} MB")
                else:
                    print(f"警告: 视频文件未生成: {self.output_path}")
            else:
                print("视频写入器已释放，但输出路径未知")
        
        cv2.destroyAllWindows()
        
        # 打印性能统计摘要
        if self.enable_perf_stats:
            self.print_perf_summary()
        
        print("播放结束")


def main():
    parser = argparse.ArgumentParser(description='BEV Video Composition and Playback Tool')
    parser.add_argument('--sample', type=int, required=True,
                       help='Sample number (e.g., 1 or 2)')
    parser.add_argument('--fo', type=int, default=0,
                       help='Front camera frame offset (default: 0)')
    parser.add_argument('--bo', type=int, default=0,
                       help='Back camera frame offset (default: 0)')
    parser.add_argument('--lo', type=int, default=0,
                       help='Left camera frame offset (default: 0)')
    parser.add_argument('--ro', type=int, default=0,
                       help='Right camera frame offset (default: 0)')
    parser.add_argument('--save', type=str, default='False',
                       choices=['True', 'False', 'true', 'false'],
                       help='Save video to file (default: False)')
    parser.add_argument('--perf', action='store_true',
                       help='Enable performance statistics (FPS, timing, etc.)')
    
    args = parser.parse_args()
    
    # 构建配置文件路径
    config_path = Path(__file__).parent / 'config.yaml'
    
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    # 设置偏移量
    offsets = {
        'front': args.fo,
        'back': args.bo,
        'left': args.lo,
        'right': args.ro
    }
    
    # 解析save参数
    save_video = args.save.lower() == 'true'
    
    # 创建并运行播放器
    player = BEVVideoPlayer(str(config_path), args.sample, offsets, save_video)
    player.enable_perf_stats = args.perf
    player.run()


if __name__ == '__main__':
    main()
