SETLOCAL

REM =============== System options
REM Define generator
SET CMGenerator="Visual Studio 16 2019"

REM Architechture: x64 or Win32
SET CMArch=x64

cmake -G %CMGenerator% -A %CMArch% -DCMAKE_BUILD_TYPE=Debug ..

ENDLOCAL
