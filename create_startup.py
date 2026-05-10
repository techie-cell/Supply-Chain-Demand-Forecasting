"""
create_startup.py
Creates batch files for easy project launch.
Run: python create_startup.py
"""
import os

root        = os.getcwd()
venv_python = os.path.join(root, "venv", "Scripts", "python.exe")
activate    = os.path.join(root, "venv", "Scripts", "activate.bat")

with open("START_API.bat", "w", encoding="utf-8") as f:
    f.write("@echo off\n")
    f.write("title API Server\n")
    f.write(f"cd /d \"{root}\"\n")
    f.write(f"call \"{activate}\"\n")
    f.write("echo Starting API on http://localhost:8000\n")
    f.write(f"\"{venv_python}\" -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000\n")
    f.write("pause\n")
print("Created: START_API.bat")

with open("START_DASHBOARD.bat", "w", encoding="utf-8") as f:
    f.write("@echo off\n")
    f.write("title Dashboard\n")
    f.write(f"cd /d \"{root}\"\n")
    f.write(f"call \"{activate}\"\n")
    f.write("echo Starting Dashboard on http://localhost:8501\n")
    f.write(f"\"{venv_python}\" -m streamlit run dashboard/app_streamlit.py --server.port 8501\n")
    f.write("pause\n")
print("Created: START_DASHBOARD.bat")

with open("START_ALL.bat", "w", encoding="utf-8") as f:
    f.write("@echo off\n")
    f.write("title AI Forecasting Launcher\n")
    f.write(f"cd /d \"{root}\"\n")
    f.write("echo Starting API Server...\n")
    f.write(f"start \"API\" cmd /c \"{root}\\START_API.bat\"\n")
    f.write("timeout /t 5 /nobreak > nul\n")
    f.write("echo Starting Dashboard...\n")
    f.write(f"start \"Dashboard\" cmd /c \"{root}\\START_DASHBOARD.bat\"\n")
    f.write("timeout /t 5 /nobreak > nul\n")
    f.write("start http://localhost:8501\n")
    f.write("pause\n")
print("Created: START_ALL.bat")

with open("TRAIN_MODELS.bat", "w", encoding="utf-8") as f:
    f.write("@echo off\n")
    f.write("title Training Models\n")
    f.write(f"cd /d \"{root}\"\n")
    f.write(f"call \"{activate}\"\n")
    f.write("echo [1/2] Training ARIMA...\n")
    f.write(f"\"{venv_python}\" run.py train --dataset M5 --model arima --max_skus 5\n")
    f.write("echo [2/2] Training XGBoost...\n")
    f.write(f"\"{venv_python}\" run.py train --dataset M5 --model xgboost --max_skus 10\n")
    f.write("echo Done! Models saved.\n")
    f.write("pause\n")
print("Created: TRAIN_MODELS.bat")

print("\n" + "="*45)
print("  All batch files created!")
print("="*45)
print("\nNext steps:")
print("  1. Double-click TRAIN_MODELS.bat")
print("  2. Double-click START_ALL.bat")
print("  3. Browser: http://localhost:8501")
