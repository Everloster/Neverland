package org.jabe.neverland.plugin;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.text.TextUtils;
import dalvik.system.DexClassLoader;

public class PluginActivity extends Activity {

	private String mPluginApkPath;
	private String mLauncherClassName;
	private IPlugin mPlugin;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		mPluginApkPath = PLUGIN_FILE_PATH;
		mLauncherClassName = "org.jabe.neverland.plugin.text.FullscreenActivity";
		copyAssertToSDCard(this, PLUGIN_FILE_PATH, PLUGIN_FILE_NAME);
		loadAPK(mPluginApkPath + File.separator + PLUGIN_FILE_NAME);
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

	private void loadAPK(String filePath) {
		try {
			final PackageInfo plocalObject = getPackageManager()
					.getPackageArchiveInfo(filePath,
							PackageManager.GET_ACTIVITIES);
			final DexClassLoader newDexClassLoader = getDexClassLoaderByPath(this,
					filePath, plocalObject.packageName);
			if ((plocalObject.activities != null)
					&& (plocalObject.activities.length > 0)) {
				String activityName = null;
				if (!TextUtils.isEmpty(mLauncherClassName)) {
					for (ActivityInfo activityInfo : plocalObject.activities) {
						if (mLauncherClassName.equals(activityInfo.name)) {
							activityName = activityInfo.name;
						}
					}
				} else {
					activityName = plocalObject.activities[0].name;
				}
				mPlugin = (IPlugin) newDexClassLoader.loadClass(activityName)
						.newInstance();
				mPlugin.onPluginCreate(filePath, this, newDexClassLoader, null);
			}
			return;
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void copyLibsToDataDir(Context context, String apkPath,
			String pluginPkgName) {
		try {
			ZipFile zipFile = new ZipFile(apkPath);
			ArrayList<ZipEntry> libEntries = new ArrayList<ZipEntry>();
			@SuppressWarnings("rawtypes")
			Enumeration e = zipFile.entries();
			while (e.hasMoreElements()) {
				ZipEntry ze = (ZipEntry) e.nextElement();
				if (ze.getName().endsWith(".so")) {
					libEntries.add(ze);
				}
			}
			for (int i = 0; i < libEntries.size(); i++) {
				ZipEntry tempEntry = libEntries.get(i);
				String[] strs = tempEntry.getName().split(File.separator);
				String fileName = strs[strs.length - 1];
				File desFile = new File(
						getPluginDataDir(context, pluginPkgName).toString()
								+ File.separator + fileName);
				InputStream in = zipFile.getInputStream(tempEntry);
				OutputStream out = new FileOutputStream(desFile);
				byte[] buf = new byte[1024];
				int len;
				while ((len = in.read(buf)) > 0) {
					out.write(buf, 0, len);
				}
				in.close();
				out.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private DexClassLoader getDexClassLoaderByPath(Context context,
			String apkPath, String pluginName) {
		final ClassLoader localClassLoader = context.getClassLoader();
		final String libPaths = getPluginDataDir(context, pluginName)
				.getAbsolutePath();
		copyLibsToDataDir(context.getApplicationContext(), apkPath,
				pluginName);
		final DexClassLoader localDexClassLoader = new DexClassLoader(apkPath,
				libPaths, libPaths, localClassLoader);
		return localDexClassLoader;
	}

	private File getPluginDataRootDir(Context context) {
		PackageInfo packageInfo = null;
		try {
			packageInfo = context.getPackageManager().getPackageInfo(
					context.getPackageName(), 0);
			String rootDirStr = packageInfo.applicationInfo.dataDir
					+ File.separator + DATA_PLUGIN_DIR_NAME;
			File rootDir = new File(rootDirStr);
			if (!rootDir.exists()) {
				rootDir.mkdirs();
			}
			return rootDir;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	private File getPluginDataDir(Context context, String pluginName) {
		final File file =  new File(getPluginDataRootDir(context), pluginName);
		if (!file.exists()) {
			file.mkdir();
		}
		return file;
	}
	
	/**
	 * Copy Assert file to sdcard
	 */
	private void copyAssertToSDCard(Context context, String desDirPath, String assetFileName) {
		try {
			new File(desDirPath).mkdirs();
			File file = new File(desDirPath, assetFileName);
			InputStream in = context.getAssets().open(assetFileName);
			file.delete();
			file.createNewFile();
			OutputStream out = new FileOutputStream(file);
			byte[] buf = new byte[1024];
			int len;
			while ((len = in.read(buf)) > 0) {
				out.write(buf, 0, len);
			}
			in.close();
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static final String ROOT_DIR_NAME = "NeverlandPlugin";
	
	public static String PLUGIN_FILE_NAME = "PluginTest.apk";
	
	public static final String DATA_PLUGIN_DIR_NAME = "plugins";

	public static String PLUGIN_FILE_PATH;
	public static String ROOT_DIR_PATH;
	static {
		ROOT_DIR_PATH = Environment.getExternalStorageDirectory()
				+ File.separator + ROOT_DIR_NAME;
		PLUGIN_FILE_PATH = ROOT_DIR_PATH + File.separator + ".plugin" + File.separator + PLUGIN_FILE_NAME;
		new File(PLUGIN_FILE_PATH).mkdirs();
	}
}
