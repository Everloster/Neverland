@echo on
set unsigned_apk_file=%project_path%\bin\%project_name%-release-unsigned.apk

::call android update project -p %project_path% -n %project_name%


::cd %project_path%
::call ant clean %1 release -Dtarget-phone=%1 -Dbuild-env=release

set signed_apk_file=E:\AutoBuildApk\%project_name%_%1_%sDate%_release_%project_version%.apk
call %local_sign%\signup.bat %unsigned_apk_file%  %signed_apk_file%
::copy %project_name%_%1_%sDate%_release_%project_version%.apk E:\AutoBuildApk\%project_name%_%1_%sDate%_release_%project_version%.apk

::call ant clean %1 release -Dtarget-phone=%1 -Dbuild-env=test
::copy %project_path%\bin\%project_name%-release-unsigned.apk E:\AutoBuildApk\%project_name%-test-unsigned.apk

echo. & pause