package com.nearme.webview.plugin;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import com.nearme.webview.plugin.util.ReflectUtil;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.KeyEvent;
import android.widget.Toast;
import dalvik.system.DexClassLoader;

public class PluginActivity extends Activity {

	private String mPluginApkPath;
	private String mLauncherClassName;
	private String mPluginLibPath;
	private IPlugin mPlugin;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		try {
			mPluginApkPath = getIntent().getStringExtra("apk_path");
			mLauncherClassName = getIntent().getStringExtra("launcher_name");
			mPluginLibPath = getIntent().getStringExtra("lib_path");
		} catch (Exception e) {
			Toast.makeText(this, "Data init error!", Toast.LENGTH_LONG).show();
			finish();
		}
		loadAPK();
	}

	@Override
	protected void onResume() {
		super.onResume();
		if (mPlugin != null) {
			mPlugin.onPluginResume();
		}
	}

	@Override
	protected void onPause() {
		super.onPause();
		if (mPlugin != null) {
			mPlugin.onPluginPause();
		}
	}

	@Override
	protected void onDestroy() {
		super.onDestroy();
		tryDestroyAssert();
		if (mPlugin != null) {
			mPlugin.onPluginDestroy();
		}
	}

	@Override
	protected void onActivityResult(int requestCode, int resultCode, Intent data) {
		super.onActivityResult(requestCode, resultCode, data);
		if (mPlugin != null) {
			mPlugin.onPluginActivityResult(requestCode, resultCode, data);
		}
	}

	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		if (mPlugin != null) {
			if (mPlugin.onPluginKeyDown(keyCode, event)) {
				return true;
			} else {
				return super.onKeyDown(keyCode, event);
			}
		} else {
			return super.onKeyDown(keyCode, event);
		}
	}

	private void loadAPK() {
		try {
			final PackageInfo plocalObject = getPackageManager()
					.getPackageArchiveInfo(mPluginApkPath,
							PackageManager.GET_ACTIVITIES);
			final DexClassLoader newDexClassLoader = getDexClassLoaderByPath();
			String activityName = null;
			if ((plocalObject.activities != null)
					&& (plocalObject.activities.length > 0)) {

				if (!TextUtils.isEmpty(mLauncherClassName)) {
					for (ActivityInfo activityInfo : plocalObject.activities) {
						if (mLauncherClassName.equals(activityInfo.name)) {
							activityName = activityInfo.name;
						}
					}
				} else {
					activityName = plocalObject.activities[0].name;
				}
			}
			mPlugin = (IPlugin) newDexClassLoader.loadClass(activityName)
					.newInstance();
			mPlugin.onPluginCreate(mPluginApkPath, this, newDexClassLoader,
					plocalObject);
			return;
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private void tryDestroyAssert() {
//		ReflectUtil.invoke(getAssets(), "destroy");
	}

	private DexClassLoader getDexClassLoaderByPath() {

		final ClassLoader localClassLoader = this.getClassLoader();

		final File libPath = new File(mPluginLibPath);
		
		if (!libPath.exists()) {
			libPath.mkdirs();
		}
		
		final DexClassLoader localDexClassLoader = new DexClassLoader(
				mPluginApkPath, mPluginLibPath, mPluginLibPath,
				localClassLoader);

		return localDexClassLoader;
	}

}
