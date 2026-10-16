"""解析命令、检查环境并分派任务。 / Parse commands, inspect the runtime, and dispatch work.

训练过程在 workflow 中；这里负责参数约束、退出码和终端提示。
Training is coordinated by workflow. This module owns CLI arguments, exit codes,
and terminal messages.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest

SCENARIOS = ('weak_signal', 'trend', 'regime_shift')
MODELS = ('mlp', 'gru', 'gnn', 'attention', 'transformer')


def verify(directory, name):
    """逐项对照清单中的文件指纹。 / Check each file against its manifest hash.

    清单只描述文件内容是否相同，不能单独证明训练或投资结论正确。
    Hashes establish file integrity; they do not establish training or investment validity.
    """
    directory = directory.resolve()
    values = json.loads((directory / name).read_text(encoding='utf-8'))['files']
    for relative, expected in values.items():
        # 清单路径必须留在待检目录内，避免相对路径跳出目录。
        # Resolve before checking containment so relative traversal is rejected.
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or Path(relative).is_absolute():
            raise ValueError('Manifest path escapes its directory')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError('SHA-256 mismatch: ' + relative)
    return dict(verified_files=len(values), scope='File integrity only / 仅文件完整性')


def main(root):
    """正常完成返回 0，已处理的运行或检查失败返回 1。
    Return 0 on success and 1 for handled execution or verification failures.

    参数解析错误与 CUDA 子进程保留各自的退出码。
    Argument parsing and the CUDA subprocess retain their own exit codes.
    """
    parser = argparse.ArgumentParser(description='Empirical Finance: three synthetic markets, optimized FP32 networks')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='Show runtime and device availability')
    commands.add_parser('verify-files', help='Verify package file hashes')
    test = commands.add_parser('test', help='Run CPU contract tests; optionally add explicit CUDA comparisons')
    test.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    run = commands.add_parser('run', help='Generate a synthetic market, fit networks and save CSV outputs')
    run.add_argument('--example', choices=(*SCENARIOS, 'all'), required=True)
    run.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    run.add_argument('--execution', choices=('optimized', 'eager'), default='optimized')
    run.add_argument('--model', choices=MODELS)
    run.add_argument('--out', type=Path, required=True)
    check = commands.add_parser('verify', help='Verify a completed run output manifest')
    check.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'doctor':
            import torch
            import numpy
            import scipy
            result = dict(python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__,
                          scipy=scipy.__version__, cuda_available=torch.cuda.is_available(),
                          cuda_runtime=torch.version.cuda,
                          device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
        elif args.command == 'verify-files':
            result = verify(root, 'MANIFEST.json')
        elif args.command == 'test':
            # CPU 检查是共同基础；--device cuda 追加五模型在同一 CUDA 设备上
            # 的常规与优化执行对照，不是比较 CPU 与 GPU 的训练结果。
            # CPU checks run first. --device cuda adds five eager/optimized comparisons
            # on the same CUDA device, not CPU-versus-GPU training comparisons.
            suite = unittest.defaultTestLoader.discover(str(root / 'tests'))
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful() or result.skipped:
                return 1
            if args.device == 'cuda':
                return subprocess.run([sys.executable, str(root / 'tests/check_cuda.py')], cwd=root).returncode
            return 0
        elif args.command == 'verify':
            # completion 表示流程走到末尾，manifest 检查落盘内容有无变化。
            # Completion records a finished workflow; the manifest checks its saved files.
            completion = json.loads((args.run / 'completion.json').read_text(encoding='utf-8'))
            if completion.get('status') != 'complete':
                raise ValueError('Run has not completed')
            result = verify(args.run, 'manifest.json')
        else:
            # 按需导入训练依赖，文件校验命令本身不需要加载 GPU 运行时。
            # Import training dependencies only for a run; file verification does not
            # load the GPU runtime.
            from .workflow import run_examples
            result = run_examples(root, list(SCENARIOS) if args.example == 'all' else [args.example],
                                  args.out, args.device, args.model, args.execution)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        print(f'Error / 错误: {exc}', file=sys.stderr)
        return 1
