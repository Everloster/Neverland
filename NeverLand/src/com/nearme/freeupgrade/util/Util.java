/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-7-25
 */
package com.nearme.freeupgrade.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import android.content.Context;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-7-25
 */
public class Util {

	/**
	 * Copy Assert file to sdcard
	 */
	public static void CopyAssertTOSD(Context context, String desDirPath, String assetFileName) {
		try {
			File file = new File(desDirPath, assetFileName);
			InputStream in = context.getAssets().open(assetFileName);
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
	 * 复制文件
	 * 
	 * @param context
	 * @param desFilePath
	 * @param originalFilePath
	 */
	public static void copyFile(Context context, String desFilePath, String originalFilePath) {
		try {
			File desFile = new File(desFilePath);
			File oriFile = new File(originalFilePath);
			InputStream in = new FileInputStream(oriFile);
			OutputStream out = new FileOutputStream(desFile);
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
}
