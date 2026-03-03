@echo off
setlocal enabledelayedexpansion
set JAVA_HOME=C:\Program Files\Java\jdk-24
set PATH=%JAVA_HOME%\bin;%PATH%
C:/Users/USER/Machine-Learning-in-Streaming/.venv/Scripts/python.exe spark/anomaly_detection.py --bootstrap-servers localhost:9092 --topic transactions --threshold 3.0
pause
