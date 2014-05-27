/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-8-2
 */
package com.nearme.freeupgrade.lib.util;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.os.Build;
import android.os.Environment;
import android.telephony.TelephonyManager;
import android.text.TextUtils;
import android.util.DisplayMetrics;
import android.view.WindowManager;
import dalvik.system.DexClassLoader;

/**
 * 存放一些仅是插件内用的工具，并管理classloader
 * 
 * @Author liu jing qiang
 * @Since 2013-8-2
 */
public class InnerUtil {

	private static final char[] HEX_DIGITS = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
			'a', 'b', 'c', 'd', 'e', 'f' };

	/**
	 * 存放不同路径插件apk存放的DexClassLoader
	 */
	private static HashMap<String, DexClassLoader> DexLoaderMap = new HashMap<String, DexClassLoader>();

	/**
	 * 根据插件apk路径获取相应的DexClassLoader
	 * 
	 * @param context
	 * @param apkPath
	 * @return
	 */
	public static DexClassLoader getDexClassLoaderByPath(Context context, String apkPath,
			String pluginPkgName) {
		long start = System.currentTimeMillis();
		DexClassLoader localDexClassLoader = DexLoaderMap.get(apkPath);
		if (localDexClassLoader == null) {
			LogUtil.i(Constants.TAG, "getDexClassLoaderByPath:" + apkPath);
			ClassLoader localClassLoader = context.getClassLoader();
			String path = getDataPluginDir(context, pluginPkgName).getAbsolutePath();
			localDexClassLoader = new DexClassLoader(apkPath, path, path, localClassLoader);
			InnerUtil.initLibs(context.getApplicationContext(), apkPath, pluginPkgName);
			DexLoaderMap.put(apkPath, localDexClassLoader);
		}
		LogUtil.i(Constants.TAG, "getDexClassLoaderByPath time:"
				+ (System.currentTimeMillis() - start));
		return localDexClassLoader;
	}

	/**
	 * 初始化apk包中的lib，将其拷贝到cache目录下
	 * 
	 * @param apkPath
	 */
	private static void initLibs(Context context, String apkPath, String pluginPkgName) {
		long start = System.currentTimeMillis();
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
				File desFile = new File(getDataPluginDir(context, pluginPkgName) + File.separator
						+ fileName);
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
		LogUtil.i(Constants.TAG, "initLibs time:" + (System.currentTimeMillis() - start));
	}

	/**
	 * 获取插件在data/data/主项目中保存信息的根位置
	 * 
	 * @param context
	 * @return
	 */
	public static File getDataPluginRootDir(Context context) {
		PackageInfo packageInfo = null;
		try {
			packageInfo = context.getPackageManager().getPackageInfo(context.getPackageName(), 0);
			String rootDirStr = packageInfo.applicationInfo.dataDir + File.separator
					+ Constants.DATA_PLUGIN_DIR_NAME;
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

	/**
	 * 根据包名获取该插件信息的保存位置
	 * 
	 * @param context
	 * @param pkgName
	 * @return
	 */
	public static File getDataPluginDir(Context context, String pkgName) {
		File dir = new File(getDataPluginRootDir(context), pkgName);
		if (!dir.exists()) {
			dir.mkdirs();
		}
		return dir;
	}

	/**
	 * <p>
	 * Get string in UTF-8 encoding
	 * </p>
	 * 
	 * @param b
	 *            byte array
	 * @return string in utf-8 encoding, or empty if the byte array is not
	 *         encoded with UTF-8
	 */
	public static String getUTF8String(byte[] b) {
		if (b == null)
			return "";
		return getUTF8String(b, 0, b.length);
	}

	/**
	 * <p>
	 * Get string in UTF-8 encoding
	 * </p>
	 */
	public static String getUTF8String(byte[] b, int start, int length) {
		if (b == null) {
			return "";
		} else {
			try {
				return new String(b, start, length, "UTF-8");
			} catch (UnsupportedEncodingException e) {
				return "";
			}
		}
	}

	/**
	 * <p>
	 * Get UTF8 bytes from a string
	 * </p>
	 * 
	 * @param string
	 *            String
	 * @return UTF8 byte array, or null if failed to get UTF8 byte array
	 */
	public static byte[] getUTF8Bytes(String string) {
		if (string == null)
			return new byte[0];

		try {
			return string.getBytes("UTF-8");
		} catch (UnsupportedEncodingException e) {
			/*
			 * If system doesn't support UTF-8, use another way
			 */

			try {
				ByteArrayOutputStream bos = new ByteArrayOutputStream();
				DataOutputStream dos = new DataOutputStream(bos);
				dos.writeUTF(string);
				byte[] jdata = bos.toByteArray();
				bos.close();
				dos.close();
				byte[] buff = new byte[jdata.length - 2];
				System.arraycopy(jdata, 2, buff, 0, buff.length);
				return buff;
			} catch (IOException ex) {
				return new byte[0];
			}
		}
	}

	public static String getIMEI(Context mContext) {
		TelephonyManager telephonyManager = (TelephonyManager) mContext
				.getSystemService(Context.TELEPHONY_SERVICE);
		String imei = telephonyManager.getDeviceId();
		if (TextUtils.isEmpty(imei)) {
			return "";
		}
		return imei;
	}

	/**
	 * 返回sd卡路径
	 * 
	 * @param ctx
	 * @return
	 */
	public static File getSdRootFile(Context ctx) {
		try {
			return android.os.Environment.getExternalStorageDirectory();
		} catch (Exception e) {
			return android.os.Environment.getExternalStorageDirectory();
		}
	}

	/**
	 * 返回下载用临时文件
	 * 
	 * @param fileName
	 * @return
	 */
	public static File getDownloadTempFile(String fileName) {
		File subFile = new File(Constants.MAIN_PROJECT_DIR_PATH + File.separator
				+ Constants.PLUGIN_DOWNLOAD_DIR_NAME + File.separator + fileName + "."
				+ Constants.SUFFIX_TEMP);
		return subFile;
	}

	/**
	 * 返回下载用临时文件
	 * 
	 * @param fileName
	 * @return
	 */
	public static void deleteDownloadTempFile(String fileName) {
		try {
			getDownloadTempFile(fileName).delete();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * 返回插件文件
	 * 
	 * @param fileName
	 * @return
	 */
	public static File getPluginFile(String fileName) {
		File subFile = new File(Constants.MAIN_PROJECT_DIR_PATH + File.separator
				+ Constants.PLUGIN_DOWNLOAD_DIR_NAME + File.separator + fileName);
		return subFile;
	}

	/**
	 * 获取插件保存路径
	 * 
	 * @return
	 */
	public static File getPluginDir() {
		File subFile = new File(Constants.MAIN_PROJECT_DIR_PATH + File.separator
				+ Constants.PLUGIN_DOWNLOAD_DIR_NAME);
		return subFile;
	}

	/**
	 * 返回sd卡是否挂载
	 * 
	 * @return
	 */
	public static boolean isSdcardExist() {
		if (Environment.MEDIA_MOUNTED.equalsIgnoreCase(Environment.getExternalStorageState())) {
			return true;
		} else {
			return false;
		}
	}

	/**
	 * 返回屏幕分辨率
	 * 
	 * @param context
	 * @return
	 */
	public static String getScreenSize(Context context) {
		DisplayMetrics dm = new DisplayMetrics();
		WindowManager manager = (WindowManager) context.getApplicationContext().getSystemService(
				Context.WINDOW_SERVICE);
		manager.getDefaultDisplay().getMetrics(dm);

		String screenSize = null;
		if (dm.heightPixels > dm.widthPixels) {
			screenSize = dm.heightPixels + "#" + dm.widthPixels;
		} else {
			screenSize = dm.widthPixels + "#" + dm.heightPixels;
		}
		return screenSize;
	}

	/**
	 * 返回手机型号
	 * 
	 * @return
	 */
	public static String getMobileName() {
		return Build.MODEL;
	}

	/**
	 * 返回sdk api版本
	 * 
	 * @return
	 */
	public static int getSdkApi() {
		return Build.VERSION.SDK_INT;
	}

	private static String toHexString(byte[] b) {
		StringBuilder sb = new StringBuilder(b.length * 2);
		for (int i = 0; i < b.length; i++) {
			sb.append(HEX_DIGITS[(b[i] & 0xf0) >>> 4]);
			sb.append(HEX_DIGITS[b[i] & 0x0f]);
		}
		return sb.toString();
	}

	/**
	 * 计算文件MD5
	 * 
	 * @param file
	 * @return
	 */
	public static String getFileMD5(File file) {
		InputStream fis;
		byte[] buffer = new byte[1024];
		int numRead = 0;
		MessageDigest md5;
		try {
			fis = new FileInputStream(file);
			md5 = MessageDigest.getInstance("MD5");
			while ((numRead = fis.read(buffer)) > 0) {
				md5.update(buffer, 0, numRead);
			}
			fis.close();
			return toHexString(md5.digest());
		} catch (Exception e) {
			LogUtil.i(Constants.TAG, "getMd5Exception:" + e.getMessage());
		} catch (OutOfMemoryError e) {
			LogUtil.i(Constants.TAG, "getMd5OutOfMemoryError:" + e.getMessage());
		}
		return null;
	}
}
