@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: 检查 .env 是否存在
if not exist ".env" (
    if exist ".env.example" (
        echo [WARN] .env not found, copying from .env.example...
        copy .env.example .env >nul
        echo [WARN] Please edit .env and set your DEEPSEEK_API_KEY
        pause
        exit /b 1
    )
)

:: 检查依赖
python -c "import textual" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt -q
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

:: 启动
python main.py
pause
