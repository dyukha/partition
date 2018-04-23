@echo off
call compile_win.cmd
set dir=output\%1
echo %dir%
mkdir %dir%
output\Runner.exe %dir% 2>&1 | tee %dir%/_err