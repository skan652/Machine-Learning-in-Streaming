@echo off
setlocal enabledelayedexpansion
set JAVA_HOME=C:\Program Files\Java\jdk-24
set PATH=%JAVA_HOME%\bin;%PATH%
C:/Users/USER/Machine-Learning-in-Streaming/.venv/Scripts/python.exe kafka/producer.py --bootstrap-servers localhost:9092 --topic transactions --csv-path data/creditcard.csv --rows 500 --delay-seconds 0.02
pause
