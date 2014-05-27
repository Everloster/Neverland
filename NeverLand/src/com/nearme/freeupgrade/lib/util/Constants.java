/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-7-25
 */
package com.nearme.freeupgrade.lib.util;

import java.io.File;

import android.os.Environment;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-7-25
 */
public class Constants {

	public static final String ROOT_DIR_NAME = "FreeUpgrade";
	public static String ROOT_DIR_PATH;
	 public static String PLUGIN_FILE_NAME = "NMWebViewPlugin.apk";
//	public final static String PLUGIN_FILE_NAME = "FreeUpgradePlugin.jar";
	// public final static String PLUGIN_FILE_NAME = "NearMeMarket.apk";
	// public final static String PLUGIN_FILE_NAME = "FWeatherSnow.apk";
	// public final static String PLUGIN_FILE_NAME = "FWeatherDande.apk";
	public static String PLUGIN_FILE_PATH;
	static {
		ROOT_DIR_PATH = Environment.getExternalStorageDirectory() + File.separator + ROOT_DIR_NAME;
		PLUGIN_FILE_PATH = Environment.getExternalStorageDirectory() + File.separator
				+ ROOT_DIR_NAME + File.separator + ".plugin" + File.separator + PLUGIN_FILE_NAME;
	}

	public final static boolean DEBUG = true;

	/*************************************** 以上可在发布是做修改 *********************************************/

	public final static String TAG = "FreeUpgrade";

	public static String MAIN_PROJECT_DIR_PATH;
	public final static String PLUGIN_DOWNLOAD_DIR_NAME = ".plugin";
	public static final String DATA_PLUGIN_DIR_NAME = "plugin";// plugin在data/data
																// 目录下存放的位置，/plugin/+插件包名

	public static final String SUFFIX_PLUGIN = "jar";// 下载文件后缀名
	public static final String SUFFIX_TEMP = "tmp";// 临时文件后缀名

	public static final String DOWNLOAD_SERVICE = "com.nearme.freeupgrade.DOWNLOAD_SERVICE";

	// 下载文件的状态
	public static final int STATUS_DOWNLOADING = 1;
	public static final int STATUS_WAITING = 2;
	public static final int STATUS_PAUSE = 3;
	public static final int STATUS_DOWNLOADED = 4;
	public static final int STATUS_CONNECTING = 5;

	public static final String EXTRA_KEY_PLUGIN_ID = "extra_key_plugin_id";
	public static final String EXTRA_KEY_PLUGIN_ITEM = "extra_key_plugin_item";
	public static final String EXTRA_KEY_START = "extra_key_start";

}
