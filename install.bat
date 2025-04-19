@echo off

:: Set Python path (adjust this if needed)
set PYTHON_EXE=python.exe

setlocal enabledelayedexpansion

:: Set current directory
cd /d %~dp0

echo Starting installation process...

:: Create and activate virtual environment
echo Creating and activating virtual environment...
%PYTHON_EXE% -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

cd code
echo Installing torch...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

echo Installing requirements...
pip install -r requirements.txt
cmd
