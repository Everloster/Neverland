#include <com_example_hellojni_HelloJni.h>

/*
 * Class:     com_example_hellojni_HelloJni
 * Method:    stringFromJNI
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_example_hellojni_HelloJni_stringFromJNI(
		JNIEnv* pEnv, jobject pThis) {
}

/*
 * Class:     com_example_hellojni_HelloJni
 * Method:    unimplementedStringFromJNI
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_example_hellojni_HelloJni_unimplementedStringFromJNI(
		JNIEnv* pEnv, jobject pThis) {
}

/*
 * Class:     com_example_hellojni_HelloJni
 * Method:    getData
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_example_hellojni_HelloJni_getData(JNIEnv* pEnv,
		jobject pThis) {
	return (*pEnv)->NewStringUTF(pEnv, "My native project talks C++");
}
