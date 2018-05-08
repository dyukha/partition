@echo off
mkdir output
g++ -Wall -Wno-sign-compare -Wextra -std=c++17 src/Runner.cpp -O2 -o output/Runner.exe 