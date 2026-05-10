@echo off
title AI Forecasting Launcher
cd /d "D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project"
echo Starting API Server...
start "API" cmd /c "D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project\START_API.bat"
timeout /t 5 /nobreak > nul
echo Starting Dashboard...
start "Dashboard" cmd /c "D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project\START_DASHBOARD.bat"
timeout /t 5 /nobreak > nul
start http://localhost:8501
pause
