@echo on
cd %local_sign%
jarsigner -storepass yb3k5cb7jdec -verbose -keystore android.keystore -signedjar %1.tmp %1 android.keystore
copy %1.tmp %2
del %1.tmp
cd %build_res_path%