@echo off
rem Builds nvdiffrast from NVlabs source against the pip-installed CUDA toolkit.
rem Needs: Visual Studio C++ tools, and the project .venv. No admin rights or system CUDA install.
setlocal
set "ROOT=%~dp0.."
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "CUDA_HOME=%ROOT%\.venv\Lib\site-packages\nvidia\cu13"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\nvvm\bin;%ROOT%\.venv\Scripts;%PATH%"
set "INCLUDE=%CUDA_HOME%\include;%CUDA_HOME%\include\cccl;%INCLUDE%"
set "DISTUTILS_USE_SDK=1"
set "TORCH_CUDA_ARCH_LIST=8.6"
set "MAX_JOBS=12"
nvcc --version
cl 2>&1 | findstr /C:"Version"
"%ROOT%\.venv\Scripts\python.exe" -m pip install --no-build-isolation --no-deps -v "%ROOT%\third_party\nvdiffrast"
exit /b %ERRORLEVEL%
