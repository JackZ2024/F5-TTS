@echo off
REM �������⻷��
if exist myenv (
    echo ���⻷��myenv�Ѿ����ڡ�
    call .\myenv\Scripts\activate
    REM ���д���Ŀ�ִ���ļ�
    python runTrain.py
    REM ��ʾ�û��������
    echo �������˳���
    pause
) else (
    REM �������⻷��
    echo ��Ҫ�������⻷��myenv...
    pause
)