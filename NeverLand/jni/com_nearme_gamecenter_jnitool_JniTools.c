#include <jni.h>
#include <com_nearme_gamecenter_jnitool_JniTools.h>
#include "md5.h"

/*
 * Class:     com_nearme_gamecenter_jnitool_JniTools
 * Method:    getFileMD5
 * Signature: (Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_nearme_gamecenter_jnitool_JniTools_getFileMD5(
		JNIEnv * env, jclass clazz, jstring filePath) {

	const char* path = (*env)->GetStringUTFChars(env, filePath, 0);
	char* md5 = MDFile(path);
	return (*env)->NewStringUTF(env, md5);

}
