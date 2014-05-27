#include <jni.h>
#include <com_nearme_gamecenter_jnitool_JniTools.h>
#include "md5.h"
#include "bsdiff.h"
#include "bspatch.h"

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

/*
 * Class:     com_nearme_gamecenter_jnitool_JniTools
 * Method:    genDiff
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_com_nearme_gamecenter_jnitool_JniTools_genDiff
  (JNIEnv *env, jclass cls,jstring old, jstring new, jstring patch) {
	int argc = 4;
	char * argv[argc];
	argv[0] = "bsdiff";
	argv[1] = (char*) ((*env)->GetStringUTFChars(env, old, 0));
	argv[2] = (char*) ((*env)->GetStringUTFChars(env, new, 0));
	argv[3] = (char*) ((*env)->GetStringUTFChars(env, patch, 0));

	int ret = bsdiff_main(argc, argv);

	(*env)->ReleaseStringUTFChars(env, old, argv[1]);
	(*env)->ReleaseStringUTFChars(env, new, argv[2]);
	(*env)->ReleaseStringUTFChars(env, patch, argv[3]);
	return ret;
}

/*
 * Class:     com_nearme_gamecenter_jnitool_JniTools
 * Method:    patch
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_com_nearme_gamecenter_jnitool_JniTools_patch
  (JNIEnv *env,jclass obj, jstring old, jstring new , jstring patch) {
	char * ch[4];
	ch[0]="bspatch";
	ch[1]=(char*)((*env)->GetStringUTFChars(env,old, 0));
	ch[2]=(char*)((*env)->GetStringUTFChars(env,new, 0));
	ch[3]=(char*)((*env)->GetStringUTFChars(env,patch, 0));

	int ret=bspatch_main(4, ch);
	(*env)->ReleaseStringUTFChars(env,old,ch[1]);
	(*env)->ReleaseStringUTFChars(env,new,ch[2]);
	(*env)->ReleaseStringUTFChars(env,patch,ch[3]);

	return ret;
}
