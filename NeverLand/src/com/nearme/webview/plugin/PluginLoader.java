package com.nearme.webview.plugin;

import java.io.File;

import org.json.JSONException;
import org.json.JSONObject;

import com.nearme.webview.plugin.util.FileUtil;
import com.nearme.webview.plugin.util.HashUtil;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.os.Handler;

public class PluginLoader implements Runnable {

	private PluginLoaderInterface mListener;

	private PluginConfig mConfig;

	private Context mContext;

	private Handler mMainThreadHandler;

	private static final String DEFAULT_ROOT_DIR_NAME = "nm_plugins";

	public PluginLoader(Context context, PluginConfig config, PluginLoaderInterface pluginLoaderInterface) {
		this.mContext = context;
		this.mConfig = config;
		this.mListener = pluginLoaderInterface;
		mMainThreadHandler = new Handler();
	}

	public void startLoading() {

		String apkPath = getApkPath();

		if (apkPath == null) {
			reLoadPluginFile();
			apkPath = getApkPath();
			if (apkPath == null) {
				// It's terrible!!
				mListener
						.onLoadFinish(PluginLoaderInterface.Result.UNKNOWN_FAIL);
				return;
			}
		}

		// here need check the apk file to ensure security
		mListener.onBeforeBackground();
		new Thread(this).start();

	}

	public void reLoadPluginFile() {

		// start copy assert file to plugin dir, file is small, so run on main
		// thread.
		FileUtil.copyAssertToSDCard(mContext, new File(getPluginRootDir(),
				mConfig.mPluginName).getAbsolutePath(), mConfig.mPluginName);

		final File libPath = getLibPath();
		
		FileUtil.removeFile(libPath, false);
		
		FileUtil.copyLibsToDataDir(mContext, getApkPath(), libPath.getAbsolutePath());

	}

	private File getLibPath() {
		return new File(getPluginRootDir(), mConfig.mPackageName);
	}

	private String getApkPath() {
		final File rootDir = getPluginRootDir();
		if (!rootDir.exists()) {
			rootDir.mkdirs();
		}
		final File realFile = new File(rootDir, mConfig.mPluginName);
		if (realFile.exists()) {
			return realFile.getAbsolutePath();
		} else {
			return null;
		}
	}

	protected File getPluginRootDir() {
		return mContext.getDir(DEFAULT_ROOT_DIR_NAME, Context.MODE_PRIVATE);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Runnable#run()
	 */
	@SuppressWarnings("static-access")
	@Override
	public void run() {

		try {

			final String path = getApkPath();
			final PackageInfo pi = mContext.getPackageManager()
					.getPackageArchiveInfo(path, 0);
			final JSONObject jsonObject = new JSONObject();

			jsonObject.put("packageName", pi.packageName);
			jsonObject.put("versionCode", pi.versionCode);
			jsonObject.put("versionName", pi.versionName);
			jsonObject.put("md5", HashUtil.md5File(new File(path)));

			boolean needUpdate = false;
			boolean isApkInvalid = true;

			// TODO send json data to server to check
			Thread.currentThread().sleep(2000);

			if (!needUpdate && isApkInvalid) {
				postSuccess();
			} else {
				// to get new apk file from server.
				Thread.currentThread().sleep(2000);

				boolean success = true;
				if (success) {
					postSuccess();
				} else {
					postNetWorkError();
				}
			}

		} catch (JSONException e) {
			e.printStackTrace();
			postUnKnownFail();
		} catch (InterruptedException e) {
			e.printStackTrace();
			postUnKnownFail();
		}

	}

	private void postSuccess() {
		mMainThreadHandler.post(new Runnable() {

			@Override
			public void run() {
				mListener.onBackgroundFinish();
				mListener.onLoadFinish(PluginLoaderInterface.Result.SUCCESS);
			}
		});
	}

	private void postNetWorkError() {
		mMainThreadHandler.post(new Runnable() {

			@Override
			public void run() {
				mListener.onBackgroundFinish();
				mListener
						.onLoadFinish(PluginLoaderInterface.Result.NETWORK_ERROR);
			}
		});
	}

	private void postUnKnownFail() {
		mMainThreadHandler.post(new Runnable() {

			@Override
			public void run() {
				mListener.onBackgroundFinish();
				mListener
						.onLoadFinish(PluginLoaderInterface.Result.UNKNOWN_FAIL);
			}
		});
	}

	public void openPlugin() {
		final Intent it = new Intent(mContext, PluginActivity.class);
		it.putExtra("apk_path", getApkPath());
		it.putExtra("lib_path", getLibPath().getAbsolutePath());
		it.putExtra("launcher_name", mConfig.mLauncherClassName);
		it.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
		mContext.startActivity(it);
	}
}
