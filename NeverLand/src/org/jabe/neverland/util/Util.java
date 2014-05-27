package org.jabe.neverland.util;

import android.content.Context;
import android.util.Log;
import android.widget.Toast;

public class Util {
	public static final int TOAST_DEFAULT_DURATION = 1000;
	private static final boolean DEVELOPMENT_LOGGING_ENABLED = true;
	private static final Object SEP_TAG = "##";
	private static final String TAG = "logcat";
	
	public static void showToast(Context ctx, String content) {
		Toast.makeText(ctx, content, TOAST_DEFAULT_DURATION).show();
	}
	
	public static void dout(String TAG, String content) {
		if (DEVELOPMENT_LOGGING_ENABLED) {
			try {
				Throwable th = new Throwable();
				StackTraceElement stack[] = th.getStackTrace();
				StackTraceElement ste = stack[1]; 
				final String className = ste.getClassName();
				final String methodName = ste.getMethodName();
				Log.v(TAG, appendLogInfo(className, methodName, content));
			} catch (Exception e) {
				
			}
		}
	}
	
	public static void dout(String content) {
		dout(TAG, content);
	}
	
	public static String appendLogInfo(String className, String method, String content) {
		return new StringBuilder(className).append(SEP_TAG).append(method).append(SEP_TAG).append(content).toString();
	}
}
