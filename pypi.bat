@echo off

echo INSTALLING PYPI LIBRARIES:
echo.

pip install torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r .\build\pypi.txt

echo.
pause
