@echo off
title Emotion Voice Studio
cd /d "%~dp0"
start "" /b cmd /c "timeout /t 6 >nul && start http://127.0.0.1:7860"
.venv\Scripts\python.exe server.py
pause
