@echo off
cd /d "%~dp0"

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
    )
)

python -c "import textual" 2>nul
if %errorlevel% neq 0 (
    pip install -r requirements.txt -q
)

python main.py
pause