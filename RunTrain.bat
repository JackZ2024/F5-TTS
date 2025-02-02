@echo off
REM 激活虚拟环境
if exist myenv (
    echo 虚拟环境myenv已经存在。
    call .\myenv\Scripts\activate
    REM 运行打包的可执行文件
    python runTrain.py
    REM 提示用户操作完成
    echo 程序已退出。
    pause
) else (
    REM 创建虚拟环境
    echo 需要创建虚拟环境myenv...
    pause
)