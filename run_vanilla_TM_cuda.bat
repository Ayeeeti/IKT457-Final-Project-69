@echo off
setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Visual Studio Installer was not found. Install Visual Studio 2022 Build Tools with the C++ build tools.
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (
    echo Visual Studio C++ Build Tools were not found.
    exit /b 1
)

set "VSDEVCMD=%VSROOT%\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" (
    echo Visual Studio developer environment was not found: "%VSDEVCMD%"
    exit /b 1
)

call "%VSDEVCMD%" -arch=amd64
if errorlevel 1 exit /b %errorlevel%

py -3.14 "%~dp0vanilla_TM.py" %*
exit /b %errorlevel%
