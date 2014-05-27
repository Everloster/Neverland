/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2013-9-11
 */
package com.nearme.webview.plugin.util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import android.content.Context;

/**
 * 
 * @Author	lailong
 * @Since	2013-9-11
 */
public class FileUtil {

	
	/**
	 * Copy Assert file to sdcard
	 * 
	 * @param context
	 * @param desDirPath
	 * @param assetFileName
	 */
	public static void copyAssertToSDCard(Context context, String desDirPath, String assetFileName) {
		try {
			File file = new File(desDirPath);
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
	
	
	/**
	 * 删除文件或者目录
	 * 
	 * @param path
	 * @param needRemoveRoot
	 */
	public static void removeFile(File path, boolean needRemoveRoot) {
		try {
			if (path.isDirectory()) {
				File[] child = path.listFiles();
				if ((child != null) && (child.length != 0)) {
					for (int i = 0; i < child.length; i++) {
						removeFile(child[i], true);
						child[i].delete();
					}
				}
			}
			if (needRemoveRoot) {
				path.delete();
			}
		} catch (Exception e) {
		}
	}
	
	public static void copyLibsToDataDir(Context context, String apkPath,
			String libPath) {
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
				File desFile = new File(libPath, fileName);
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
}
