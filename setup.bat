@echo off
REM ����Ƿ��Ѿ���һ����Ϊmyenv�����⻷��
if exist myenv (
    echo ���⻷��myenv�Ѿ����ڡ�
    pause
) else (
    REM �������⻷��
    echo �������⻷��myenv...
    python -m venv myenv
    REM �������⻷������װ������
    call .\myenv\Scripts\activate
    pip cache purge
	REM pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
	pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -e .
	pip install gdown
	pip install tensorboard
    echo ���⻷��������������ɡ�
    pause
    deactivate
)