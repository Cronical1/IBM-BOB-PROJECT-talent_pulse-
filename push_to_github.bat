@echo off
setlocal
echo ===================================================================
echo   TalentPulse AI - GitHub Push Helper
echo ===================================================================
echo.
set /p REPO_URL="Enter your GitHub Repository HTTPS URL (e.g., https://github.com/username/talent-pulse.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] No URL provided. Exiting...
    pause
    exit /b 1
)

echo.
echo [1/3] Adding remote origin...
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%

echo [2/3] Setting branch to main...
git branch -M main

echo [3/3] Pushing to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================================
    echo   [SUCCESS] TalentPulse AI has been pushed to your GitHub!
    echo   Visit your repository: %REPO_URL%
    echo ===================================================================
) else (
    echo.
    echo [NOTE] If git asked for authentication, sign in with your browser
    echo or use a GitHub Personal Access Token (PAT).
)
echo.
pause
