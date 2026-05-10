@echo off
title Dashboard
cd /d "D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project"
call "D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project\venv\Scripts\activate.bat"
echo Starting Dashboard on http://localhost:8501
"D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project\venv\Scripts\python.exe" -m streamlit run dashboard/app_streamlit.py --server.port 8501
pause
