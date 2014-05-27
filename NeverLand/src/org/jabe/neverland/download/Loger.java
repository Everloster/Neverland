package org.jabe.neverland.download;

public class Loger {

	public interface ILog {
		public void log(String tag, String message, int level);
		public void log(String tag, String message);
	}

	private volatile static ILog instance;

	public static void log(String TAG, String message, int level) {
		if (instance == null) {
			instance = getDefaultLoger();
		}
		instance.log(TAG, message, level);
	}

	public static void log(String TAG, String message) {
		if (instance == null) {
			instance = getDefaultLoger();
		}
		instance.log(TAG, message);
	}

	public static void setInstance(ILog i) {
		instance = i;
	}

	public static ILog getDefaultLoger() {
		return new ILog() {
			@Override
			public void log(String tag, String message, int level) {
				System.out.println(tag + "###" + message);
			}

			@Override
			public void log(String tag, String message) {
				System.out.println(tag + "###" + message);
			}
		};
	}
}
