@echo off
cd /d %~dp0
python -m pip install -r requirements.txt
echo.
echo Starting Everyday AI at http://127.0.0.1:8000
python server.py
pause
