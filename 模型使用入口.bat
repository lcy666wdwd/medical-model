@echo off
pushd "%~dp0apnea-ecg-test-label"
python predict.py
popd
pause
