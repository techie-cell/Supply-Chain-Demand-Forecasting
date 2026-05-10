@echo off
title API Server
cd /d "D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project"
call "D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project\venv\Scripts\activate.bat"
echo Starting API on http://localhost:8000
"D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project\venv\Scripts\python.exe" -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
pause
