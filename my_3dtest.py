# full_version_check.py（适配PyTorch 1.12.1）
import torch
import platform
import subprocess
import os

def run_command(cmd):
    """执行终端命令并返回输出"""
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return result.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        return f"命令执行失败: {e.output.decode('utf-8').strip()}"

def check_all_versions():
    print("=== 🔍 全链路版本检测报告 ===")
    
    # 1. 系统/环境基础信息
    print("\n【1. 系统与WSL信息】")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"WSL版本检测（Windows PowerShell命令，WSL内执行会报错，仅参考）:")
    print(run_command("uname -a"))
    
    # 2. NVIDIA驱动/CUDA信息
    print("\n【2. NVIDIA驱动与CUDA信息】")
    print("nvidia-smi输出:")
    nvidia_smi = run_command("nvidia-smi")
    print(nvidia_smi[:500] + "..." if len(nvidia_smi) > 500 else nvidia_smi)
    
    print("\n系统CUDA版本（WSL侧安装的CUDA Toolkit）:")
    print(run_command("nvcc --version | grep release || echo '未安装nvcc'"))
    
    # 3. PyTorch深度版本信息
    print("\n【3. PyTorch详细版本】")
    print(f"PyTorch基础版本: {torch.__version__}")
    print(f"PyTorch编译CUDA版本: {torch.version.cuda}")
    print(f"PyTorch cuDNN版本: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
    # 注释掉PyTorch 1.12.1不支持的行
    # print(f"PyTorch是否为CUDA构建: {torch._C._cuda_is_built()}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"PyTorch安装路径: {torch.__file__}")
    
    # 4. 环境变量检查
    print("\n【4. 关键环境变量】")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '未设置')}")
    print(f"PATH中的CUDA路径: {[p for p in os.environ.get('PATH','').split(':') if 'cuda' in p.lower()]}")
    
    # 5. Conda环境信息
    print("\n【5. Conda虚拟环境】")
    print(f"当前Conda环境: {os.environ.get('CONDA_DEFAULT_ENV', '未激活Conda')}")
    print("Conda已安装包（PyTorch相关）:")
    print(run_command("conda list | grep -E 'torch|cuda' || echo 'Conda未找到相关包'"))

if __name__ == "__main__":
    # 配置WSL GPU库路径
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    check_all_versions()