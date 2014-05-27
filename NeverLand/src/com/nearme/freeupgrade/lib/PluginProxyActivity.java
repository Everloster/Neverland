/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-7-25
 */
package com.nearme.freeupgrade.lib;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.text.TextUtils;

import com.nearme.freeupgrade.lib.util.Constants;
import com.nearme.freeupgrade.lib.util.InnerUtil;
import com.nearme.freeupgrade.lib.util.LogUtil;

import dalvik.system.DexClassLoader;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-7-25
 */
public class PluginProxyActivity extends Activity {
	public static final String EXTRA_PLUGIN_APKPATH = "plugin.Location";
	public static final String EXTRA_LAUNCH_ACTIVITY = "plugin.launchActivity";

	private String mPluginLoaction;
	private String mLaunchActivity;
	private IActivity mPlugin;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		Intent localIntent = getIntent();
		mPluginLoaction = localIntent.getStringExtra(EXTRA_PLUGIN_APKPATH);
		mLaunchActivity = localIntent.getStringExtra(EXTRA_LAUNCH_ACTIVITY);
		if(TextUtils.isEmpty(mPluginLoaction)){
			finish();
			return;
		}
		loadAPK(mPluginLoaction);
	}

	@Override
	protected void onResume() {
		super.onResume();
		if (mPlugin != null) {
			mPlugin.IOnResume();
		}
	}

	@Override
	protected void onPause() {
		super.onPause();
		if (mPlugin != null) {
			mPlugin.IOnPause();
		}
	}

	@Override
	protected void onDestroy() {
		super.onDestroy();
		if (mPlugin != null) {
			mPlugin.IOnDestroy();
		}
	}
	
	@Override
	protected void onActivityResult(int requestCode, int resultCode, Intent data) {
		// TODO Auto-generated method stub
		super.onActivityResult(requestCode, resultCode, data);
		if (mPlugin != null) {
//			mPlugin.Iona
		}
	}

	public void loadAPK(String filePath) {
		LogUtil.i(Constants.TAG, "apkPath:" + filePath);
		try {
			PackageInfo plocalObject = getPackageManager().getPackageArchiveInfo(filePath,
					PackageManager.GET_ACTIVITIES);
			DexClassLoader localDexClassLoader = InnerUtil.getDexClassLoaderByPath(this, filePath,
					plocalObject.packageName);
			if ((plocalObject.activities != null) && (plocalObject.activities.length > 0)) {
				String activityName = null;
				if (!TextUtils.isEmpty(mLaunchActivity)) {
					for (ActivityInfo activityInfo : plocalObject.activities) {
						if (mLaunchActivity.equals(activityInfo.name)) {
							LogUtil.i(Constants.TAG, "activityname = " + activityInfo.name);
							activityName = activityInfo.name;
						}
					}
				} else {
					activityName = plocalObject.activities[0].name;
				}
				mPlugin = (IActivity) localDexClassLoader.loadClass(activityName).newInstance();
				mPlugin.IInit(filePath, this, localDexClassLoader, null);
				mPlugin.ISetIntent(getIntent());
				mPlugin.IOnCreate(null);
			}
			return;
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}
