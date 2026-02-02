#!/usr/bin/env python3
"""
性能分析脚本 - 使用cProfile分析main.py的性能
"""

import cProfile
import pstats
import io
import sys
from pathlib import Path

# 导入main模块
import main

def profile_main():
    """使用cProfile分析main.py的性能"""
    # 创建性能分析器
    profiler = cProfile.Profile()
    
    # 开始分析
    profiler.enable()
    
    # 运行main函数（需要模拟命令行参数）
    # 注意：这里需要根据实际情况调整参数
    sys.argv = ['profile_main.py', '--sample', '1', '--save', 'False']
    
    try:
        main.main()
    except KeyboardInterrupt:
        print("\n性能分析被中断")
    finally:
        profiler.disable()
    
    # 生成统计报告
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')  # 按累计时间排序
    ps.print_stats(50)  # 打印前50个最耗时的函数
    
    # 保存到文件
    output_file = Path(__file__).parent / 'performance_profile.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(s.getvalue())
    
    print(f"\n性能分析报告已保存到: {output_file}")
    print("\n=== 性能分析摘要 ===")
    print(s.getvalue()[:2000])  # 打印前2000个字符

if __name__ == '__main__':
    print("开始性能分析...")
    print("注意：程序会运行完整的视频处理流程，按Ctrl+C可以中断")
    profile_main()
