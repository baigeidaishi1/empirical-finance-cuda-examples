"""从命令行启动合成实验。 / Start synthetic experiments from the command line.

本文件定位源码并交给 cli 解析命令；数据生成、训练和回测由各自模块负责。
This file locates the source and delegates argument parsing to cli. Separate
modules handle data generation, training, and portfolio accounting.
"""
import os
from pathlib import Path
import sys

# 在导入 PyTorch 前设置 cuBLAS 工作区，配合训练模块中的确定性计算设置。
# setdefault 会保留调用者已经指定的值。
# Set the cuBLAS workspace before importing PyTorch, alongside the engine's
# deterministic settings. setdefault preserves an explicit caller setting.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parent
# 用脚本位置定位 src；用户的输出路径仍相对于运行命令时的工作目录。
# Locate src from this script; user-supplied output paths remain relative to cwd.
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    from efcuda.cli import main
    raise SystemExit(main(ROOT))
