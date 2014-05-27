/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2013-9-4
 */
package com.nearme.plugin.webview.client.cordova;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;

import org.apache.cordova.CallbackContext;
import org.apache.cordova.CordovaPlugin;
import org.apache.cordova.PluginResult;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.net.Uri;

/**
 * 
 * @Author	lailong
 * @Since	2013-9-4
 */
public class AppLauncher extends CordovaPlugin {
	
	private static final String RESULT_STRING_NULL = "null";
	private static final String RESULT_STRING_OK = "ok";
	private static final String RESULT_STRING_FAIL = "fail";
	
	/* (non-Javadoc)
	 * @see org.apache.cordova.CordovaPlugin#execute(java.lang.String, org.json.JSONArray, org.apache.cordova.CallbackContext)
	 */
	@Override
	public boolean execute(String action, JSONArray args,
			final CallbackContext callbackContext) throws JSONException {
        try {
        	if (action.equals("app-open")) {
        		final String packageName = args.getString(0);
        		openGame(packageName, callbackContext);
        	} else if (action.equals("app-get-info")) {
        		final String pkgName = args.getString(0);
        		final PackageInfo pi = cordova.getActivity().getPackageManager().getPackageInfo(pkgName, 0);
        		if (pi != null) {
        			final JSONObject jsonResult = getJsonByPackageInfo(pi);
        			callbackContext.sendPluginResult(new PluginResult(PluginResult.Status.OK, jsonResult));
        		} else {
        			callbackContext.sendPluginResult(new PluginResult(PluginResult.Status.OK, RESULT_STRING_NULL));
        		}
        	} else if (action.equals("app-get-all-info")) {
        		final Runnable runnable = new Runnable() {
					
					@Override
					public void run() {
						try {
							final List<PackageInfo> apps = cordova.getActivity().getPackageManager().getInstalledPackages(PackageManager.GET_SIGNATURES);
							final JSONArray jsonArray = new JSONArray();
							for (PackageInfo packageInfo : apps) {
								// 屏蔽系统应用
								if ((packageInfo.applicationInfo.flags & ApplicationInfo.FLAG_SYSTEM) <= 0) {
									final JSONObject jsonObject = getJsonByPackageInfo(packageInfo);
									jsonArray.put(jsonObject);
								}
							}
							callbackContext.success(jsonArray);
						} catch (Exception e) {
							callbackContext.success(RESULT_STRING_NULL);
						}
					}
				};
				cordova.getThreadPool().execute(runnable);
        	} else if (action.equals("app-install")) {
        		final String apkPath = args.getString(0);
        		if (apkPath != null && !apkPath.equals("")) {
        			final boolean needAutoInstall = args.getBoolean(1);
        			if (needAutoInstall) {
        				tryAutoInstall(apkPath, callbackContext);
        			} else {
        				showSystemInstall(apkPath);
        			}
        		} else {
        			throw new JSONException("invalid apk path");
        		}
        	} else if (action.equals("app-intent")) {
        		
        	}
            return true;
		} catch (Exception e) {
			return false;
		}
	}

	private JSONObject getJsonByPackageInfo(final PackageInfo pi)
			throws JSONException {
		final JSONObject jsonResult = new JSONObject();
		jsonResult.put("packageName", pi.packageName);
		jsonResult.put("versionCode", pi.versionCode);
		jsonResult.put("versionName", pi.versionName);
		jsonResult.put("sourceDir", pi.applicationInfo.sourceDir);
		return jsonResult;
	}
	
	private void showSystemInstall(final String apkPath) {
		File localFile = new File(apkPath);
		Intent intent = new Intent(Intent.ACTION_VIEW);
		intent.setDataAndType(Uri.fromFile(localFile), "application/vnd.android.package-archive");
		intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
		cordova.getActivity().startActivity(intent);
	}	
	
	private void openGame(final String packageName, final CallbackContext callbackContext) {
		PackageManager packageManager = cordova.getActivity().getPackageManager();
		Intent intent = new Intent();
		try {
			intent = packageManager.getLaunchIntentForPackage(packageName);
			intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
			cordova.getActivity().startActivity(intent);
			callbackContext.success(RESULT_STRING_OK);
		} catch (Exception e) {
			e.printStackTrace();
			callbackContext.error(RESULT_STRING_FAIL);
		}
	}
	
	private void tryAutoInstall(final String localFile, final CallbackContext callbackContext) {
		
		cordova.getThreadPool().execute(new Runnable() {

			@Override
			public void run() {

				StringBuilder sb = new StringBuilder("pm install ");
				sb.append("-r ");
				sb.append(localFile);
				InstallResult result = InstallResult.OTHER_FAILED;
				Process process = null;
				try {
					process = Runtime.getRuntime().exec(sb.toString());
					InputStream is = process.getInputStream();
					InputStream eis = process.getErrorStream();
					BufferedReader reader = new BufferedReader(new InputStreamReader(is));
					BufferedReader eReader = new BufferedReader(new InputStreamReader(eis));
					String error = null;
					String tmp;
					for (;;) {
						tmp = eReader.readLine();
						if (tmp == null) {
							break;
						}
						error = tmp;
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

					if ("Success".equalsIgnoreCase(standard)) {
						result = InstallResult.SUCCESS;
					} else if (error != null
							&& error.contains("INSTALL_PARSE_FAILED_INCONSISTENT_CERTIFICATES")) {
						result = InstallResult.INSTALL_PARSE_FAILED_INCONSISTENT_CERTIFICATES;
					} else if (error != null
							&& error.contains("INSTALL_FAILED_INSUFFICIENT_STORAGE")) {
						result = InstallResult.INSTALL_FAILED_INSUFFICIENT_STORAGE;
					}
				} catch (Exception e) {
					e.printStackTrace();
				} finally {
					if (process != null) {
						process.destroy();
					}
				}
				callbackContext.sendPluginResult(new PluginResult(PluginResult.Status.OK, result.index()));
			}
		});

	}
	
	public static enum InstallResult {
		
		SUCCESS(1), //
		INSTALL_PARSE_FAILED_INCONSISTENT_CERTIFICATES(2), //
		INSTALL_FAILED_INSUFFICIENT_STORAGE(3), //
		OTHER_FAILED(4); //

		private int index;

		private InstallResult(int index) {
			this.index = index;
		}

		public int index() {
			return index;
		}

		public static InstallResult valueOf(int index) {
			switch (index) {
			case 1:
				return SUCCESS;
			case 2:
				return INSTALL_PARSE_FAILED_INCONSISTENT_CERTIFICATES;
			case 3:
				return INSTALL_FAILED_INSUFFICIENT_STORAGE;
			default:
				return OTHER_FAILED;
			}
		}
	}
}
