/**
 * Copyright (C) 2014, all rights reserved.
 * Company	Alibaba Group. 
 * Author	LaiLong
 * Since	Jul 28, 2014
 */
package org.jabe.neverland.autoinstall;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;

import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;

/**
 * 
 * 静默安装的工具类
 * 
 * @Author LaiLong
 * @Since Jul 28, 2014
 */
public class AutoInstallHelper {

	public static Map<Integer, String> sInstallErrorInfoMap = new HashMap<Integer, String>();
	
	static {
		initInstallErrorString();
	}

	private static void initInstallErrorString() {
		Field[] fields = PackageManager.class.getFields();
		for (Field f : fields) {
			if (f.getType() == int.class) {
				int modifiers = f.getModifiers();
				// only look at public final static fields.
				if (((modifiers & Modifier.FINAL) != 0) && ((modifiers & Modifier.PUBLIC) != 0)
						&& ((modifiers & Modifier.STATIC) != 0)) {
					String fieldName = f.getName();
					if (fieldName.startsWith("INSTALL_FAILED_")
							|| fieldName.startsWith("INSTALL_PARSE_FAILED_")) {
						// get the int value and store it to map.
						try {
							sInstallErrorInfoMap.put(f.getInt(null), f.getName());
						} catch (IllegalArgumentException e) {
							e.printStackTrace();
						} catch (IllegalAccessException e) {
							e.printStackTrace();
						}
					}
				}
			}
		}
	}

	private static int getInstallErrorCodeByMessage(String message) {
		final Set<Integer> keys = sInstallErrorInfoMap.keySet();
		for (Integer key : keys) {
			if (message.contains(sInstallErrorInfoMap.get(key))
					|| sInstallErrorInfoMap.get(key).equals(message)) {
				return key;
			}
		}
		return INSTALL_FAILED_UNKNOW;
	}
	
	public static String getInstallResultByCode(int code) {
		final String result = sInstallErrorInfoMap.get(code); 
		if (result != null) {
			return result;
		} else {
			return "";
		}
	}

	public static void autoInstallAsync(final File file, final InstallCallback callback,
			final ExecutorService executer) {
		
		final Handler mainHandler = new Handler(Looper.getMainLooper()) {
			
			@Override
			public void dispatchMessage(Message msg) {
				if (msg != null) {
					int code = msg.what;
					if (code < 0) {
						String message = msg.getData().getString("message");
						callback.onFailure(code, message);
					} else {
						callback.onSuccess();
					}
				} else {
					callback.onSuccess();
				}
			}
		};
		
		if (file == null || !file.isFile() || !file.exists() || callback == null) {
			if (callback != null) {
				sendFailureToHandler(INSTALL_FAILED_UNKNOW, "Invaild input file!", mainHandler);
			}
		} else {
			final Runnable worker = new Runnable() {

				@Override
				public void run() {
					StringBuilder sb = new StringBuilder("pm install ");
					sb.append("-r ");
					sb.append("-l ");
					sb.append(file.getAbsolutePath());
					Process process = null;
					BufferedReader reader = null;
					BufferedReader eReader = null;
							
					try {
						process = Runtime.getRuntime().exec(sb.toString());
						InputStream is = process.getInputStream();
						InputStream eis = process.getErrorStream();
						reader = new BufferedReader(new InputStreamReader(is));
						eReader = new BufferedReader(new InputStreamReader(eis));
						String error = null;
						String tmp;
						for (;;) {
							tmp = eReader.readLine();
							if (tmp == null) {
								break;
							}
							error = error + tmp;
						}
						String standard = null;
						tmp = null;
						for (;;) {
							tmp = reader.readLine();
							if (tmp == null) {
								break;
							}
							standard = tmp;
						}

						/**
						 * for some fucking devices
						 */
						if (error != null && error.contains("pkg") && standard == null
								&& process.exitValue() == 9) {
							// can't do auto install
							if (callback != null) {
								sendFailureToHandler(INSTALL_FAILED_UNKNOW, error, mainHandler);
							}

						} else {

							if ("Success".equalsIgnoreCase(standard)) {
								sendOkToHandler(1, mainHandler);
							} else {
								if (error != null) {
									final int code = getInstallErrorCodeByMessage(error);
									if (code == INSTALL_FAILED_UNKNOW) {
										if (callback != null) {
											sendFailureToHandler(code, "unknow error", mainHandler);
										} 
									} else {
										if (callback != null) {
											sendFailureToHandler(code, sInstallErrorInfoMap.get(code), mainHandler);
										}
									}
								} else {
									
								}
//								if (error != null
//										&& error.contains("INSTALL_PARSE_FAILED_INCONSISTENT_CERTIFICATES")) {
//								} else if (error != null
//										&& error.contains("INSTALL_FAILED_INSUFFICIENT_STORAGE")) {
//								} else if (error != null
//										&& error.contains("INSTALL_PARSE_FAILED_NO_CERTIFICATES")) {
//								}
							}

						}
					} catch (Exception e) {
						sendFailureToHandler(INSTALL_FAILED_UNKNOW, e.getMessage(), mainHandler);
					} finally {
						if (reader != null) {
							try {
								reader.close();
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
						
						if (eReader != null) {
							try {
								eReader.close();
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
						
						if (process != null) {
							process.destroy();
						}
					}
				}

			};
			executer.execute(worker);
		}
	}
	
	private static final void sendFailureToHandler(final int code, final String message, final Handler handler) {
		final Message msg = handler.obtainMessage();
		final Bundle b = new Bundle();
		b.putString("message", message);
		msg.setData(b);
		msg.what = code;
		msg.sendToTarget();
	}
	
	private static final void sendOkToHandler(final int code, final Handler handler) {
		final Message msg = handler.obtainMessage();
		msg.what = code;
		msg.sendToTarget();
	}

	public static final int INSTALL_FAILED_UNKNOW = -Integer.MIN_VALUE;

	public static interface InstallCallback {
		public void onSuccess();

		/**
		 * 安装失败的回调
		 * 
		 * @param code
		 *            一般情况和PM的code一致，INSTALL_FAILED_UNKNOW为未知。
		 * @param message
		 *            INSTALL_PARSE_FAILED_* or INSTALL_FAILED_*
		 */
		public void onFailure(final int code, final String message);
	}
}
