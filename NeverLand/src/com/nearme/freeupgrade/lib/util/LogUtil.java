/**
 * 
 */
package com.nearme.freeupgrade.lib.util;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;


import android.util.Log;

/**
 * @author 80054358
 * 
 */
public class LogUtil {
	public static void i(String tag, double info) {
		if (Constants.DEBUG) {
			StackTraceElement[] stack = (new Throwable()).getStackTrace();
			StackTraceElement ele = null;
			String className = "";
			if (stack.length > 1) {
				ele = stack[1];
				try {
					className = Class.forName(ele.getClassName()).getSimpleName();
				} catch (ClassNotFoundException e) {
				}
			}
			Log.i(tag, className + "::" + String.valueOf(info));
		}
	}

	public static void i(String tag, float info) {
		if (Constants.DEBUG) {
			StackTraceElement[] stack = (new Throwable()).getStackTrace();
			StackTraceElement ele = null;
			String className = "";
			if (stack.length > 1) {
				ele = stack[1];
				try {
					className = Class.forName(ele.getClassName()).getSimpleName();
				} catch (ClassNotFoundException e) {
				}
			}
			Log.i(tag, className + "::" + String.valueOf(info));
		}
	}

	public static void i(String tag, int info) {
		if (Constants.DEBUG) {
			StackTraceElement[] stack = (new Throwable()).getStackTrace();
			StackTraceElement ele = null;
			String className = "";
			if (stack.length > 1) {
				ele = stack[1];
				try {
					className = Class.forName(ele.getClassName()).getSimpleName();
				} catch (ClassNotFoundException e) {
				}
			}
			Log.i(tag, className + "::" + String.valueOf(info));
		}
	}

	public static void i(String tag, long info) {
		if (Constants.DEBUG) {
			StackTraceElement[] stack = (new Throwable()).getStackTrace();
			StackTraceElement ele = null;
			String className = "";
			if (stack.length > 1) {
				ele = stack[1];
				try {
					className = Class.forName(ele.getClassName()).getSimpleName();
				} catch (ClassNotFoundException e) {
				}
			}
			Log.i(tag, className + "::" + String.valueOf(info));
		}
	}

	public static void i(String tag, String info) {
		if (Constants.DEBUG) {
			StackTraceElement[] stack = (new Throwable()).getStackTrace();
			StackTraceElement ele = null;
			String className = "";
			if (stack.length > 1) {
				ele = stack[1];
				try {
					className = Class.forName(ele.getClassName()).getSimpleName();
				} catch (ClassNotFoundException e) {
				}
			}
			Log.i(tag, className + "::" + info);
		}
	}

	@SuppressWarnings("rawtypes")
	public static void i(String tag, String startMsg, Collection info) {
		if (Constants.DEBUG) {
			StackTraceElement[] stack = (new Throwable()).getStackTrace();
			StackTraceElement ele = null;
			String className = "";
			if (stack.length > 1) {
				ele = stack[1];
				try {
					className = Class.forName(ele.getClassName()).getSimpleName();
				} catch (ClassNotFoundException e) {
				}
			}
			StringBuffer msg = null;
			if (startMsg == null) {
				msg = new StringBuffer(className + "::");
			} else {
				msg = new StringBuffer(className + "::" + startMsg);
			}
			Iterator itr = info.iterator();
			while (itr.hasNext()) {
				msg.append(itr.next().toString());
				msg.append(";");
			}
			Log.i(tag, msg.toString());
		}
	}

	@SuppressWarnings("rawtypes")
	public static void i(String tag, String startMsg, HashMap info) {
		if (Constants.DEBUG) {
			StackTraceElement[] stack = (new Throwable()).getStackTrace();
			StackTraceElement ele = null;
			String className = "";
			if (stack.length > 1) {
				ele = stack[1];
				try {
					className = Class.forName(ele.getClassName()).getSimpleName();
				} catch (ClassNotFoundException e) {
				}
			}
			StringBuffer msg = null;
			if (startMsg == null) {
				msg = new StringBuffer(className + "::");
			} else {
				msg = new StringBuffer(className + "::" + startMsg);
			}
			Set keySet = info.keySet();
			Iterator itr = keySet.iterator();
			while (itr.hasNext()) {
				Object key = itr.next();
				Object value = info.get(key);
				msg.append(key.toString());
				msg.append("=");
				msg.append(value.toString());
				msg.append(";");
			}
			Log.i(tag, msg.toString());
		}
	}

}
