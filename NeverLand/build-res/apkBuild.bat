@echo on

if not exist %signed_apk_path%\%1 mkdir %signed_apk_path%\%1
set temp_signed_apk_path=%signed_apk_path%\%1

::do release build
call android update project -p %project_path% -n %project_name% -t 1
cd %project_path%
call ant clean %1 init svn-update-revert build-env-release release -Dtarget-phone=%1 -Dbuild-env=release -Dsvn_home=%svn_home%
set signed_apk_file=%temp_signed_apk_path%\%project_name%_%1_%sDate%_release_%project_version%.apk
call %local_sign%\signup.bat %unsigned_apk_file%  %signed_apk_file%


::do test build
call android update project -p %project_path% -n %project_name% -t 1
cd %project_path%
call ant clean %1 init svn-update-revert build-env-test release -Dtarget-phone=%1 -Dbuild-env=test -Dsvn_home=%svn_home%
set signed_apk_file=%temp_signed_apk_path%\%project_name%_%1_%sDate%_test_%project_version%.apk
call %local_sign%\signup.bat %unsigned_apk_file%  %signed_apk_file%

::do gamma build
call android update project -p %project_path% -n %project_name% -t 1
cd %project_path%
call ant clean %1 init svn-update-revert build-env-gamma release -Dtarget-phone=%1 -Dbuild-env=gamma -Dsvn_home=%svn_home%
set signed_apk_file=%temp_signed_apk_path%\%project_name%_%1_%sDate%_gamma_%project_version%.apk
call %local_sign%\signup.bat %unsigned_apk_file%  %signed_apk_file%