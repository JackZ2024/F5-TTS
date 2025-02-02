@echo off
REM 检查是否已经有一个名为myenv的虚拟环境
if exist myenv (
    echo 虚拟环境myenv已经存在。
    pause
) else (
    REM 创建虚拟环境
    echo 创建虚拟环境myenv...
    python -m venv myenv
    REM 激活虚拟环境并安装依赖项
    call .\myenv\Scripts\activate
    pip cache purge
	REM pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
	pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -e .
	pip install gdown
	pip install tensorboard
    echo 虚拟环境创建并配置完成。
    pause
    deactivate
)