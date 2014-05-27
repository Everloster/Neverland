@echo on
::set project_version=3.0.2
set project_name=GameCenter
::set project_path=F:\NearMeProject\GameCenter\apk\GameCenter-V3.0.2
set svn_home=C:\Program Files\Subversion\bin

set build_res_path=%project_path%\build-res
set local_sign=%project_path%\build-res\nearme_sign
set unsigned_apk_file=%project_path%\bin\%project_name%-release-unsigned.apk
set signed_apk_root=F:\APK
set signed_apk_path=%signed_apk_root%\%date:~0,10%

set sDate=%date:~0,10%
if not exist %signed_apk_root% mkdir %signed_apk_root%
if not exist %signed_apk_path% mkdir %signed_apk_path%
cd %build_res_path%