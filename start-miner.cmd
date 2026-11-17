@echo off
rem pcsolo - one-click start with the best settings measured on the dev PC:
rem   --gpu            mine on the GPU as well as the CPU
rem   --version-roll   8 header versions per nonce (+13% GPU)
rem   CPU threads are auto-tuned at start (~25 s), second-hash precompute is on by default.
rem Stop with Ctrl+C: the miner prints a summary, then answer Y to close.
rem
rem Put your own Bitcoin address below (public address only - never a seed or private key).

title pcsolo miner
set "MINER=%~dp0pcsolo.exe"
set "ADDRESS=YOUR_BTC_ADDRESS"

if "%ADDRESS%"=="YOUR_BTC_ADDRESS" (
    echo Edit start-miner.cmd and set ADDRESS to your Bitcoin address first.
    pause
    exit /b 1
)
if not exist "%MINER%" (
    echo Cannot find %MINER% - run build.cmd first.
    pause
    exit /b 1
)

"%MINER%" --address %ADDRESS% --gpu --version-roll

echo.
echo Miner stopped. Press any key to close this window.
pause >nul
