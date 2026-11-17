@echo off
rem Builds pcsolo.exe with MSYS2's UCRT64 gcc. Output is a static exe (no DLLs), runs on any x64 Windows 10/11.
set PATH=C:\msys64\ucrt64\bin;%PATH%
gcc -O3 -Wall -static -o "%~dp0pcsolo.exe" "%~dp0pcsolo.c" -lws2_32 -lwinmm
