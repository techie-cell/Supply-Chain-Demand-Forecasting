@echo off
title Training Models
cd /d "D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project"
call "D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project\venv\Scripts\activate.bat"
echo [1/2] Training ARIMA...
"D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project\venv\Scripts\python.exe" run.py train --dataset M5 --model arima --max_skus 5
echo [2/2] Training XGBoost...
"D:\AIML\Intership Project\demand_forecasting_v3\demand_forecasting_project\venv\Scripts\python.exe" run.py train --dataset M5 --model xgboost --max_skus 10
echo Done! Models saved.
pause
