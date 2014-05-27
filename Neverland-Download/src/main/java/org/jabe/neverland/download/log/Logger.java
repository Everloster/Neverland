package org.jabe.neverland.download.log;

public class Logger {

	static {
		synchronized (Logger.class) {
			sLogger = new DefaultLogger();
		}
	}
	
	private volatile static ILogger sLogger;
	private volatile static boolean sEnable = true;

	public static void i(String tag, String message) {
		if (sEnable) {
			sLogger.i(tag, message);
		}
	}

	public static void i(String message) {
		if (sEnable) {
			sLogger.i(message);
		}
	}

	public static void d(String tag, String message) {
		if (sEnable) {
			sLogger.d(tag, message);
		}
	}

	public static void d(String message) {
		if (sEnable) {
			sLogger.d(message);
		}
	}

	public static void w(String tag, String message) {
		if (sEnable) {
			sLogger.w(tag, message);
		}
	}

	public static void w(String message) {
		if (sEnable) {
			sLogger.w(message);
		}
	}

	public static void e(String tag, String message) {
		if (sEnable)	{
			sLogger.e(tag, message);
		}
	}

	public static void e(String message) {
		if (sEnable) {
			sLogger.e(message);
		}
	}
	
	public static void enable(boolean enable) {
		sEnable = enable;
	}

}
