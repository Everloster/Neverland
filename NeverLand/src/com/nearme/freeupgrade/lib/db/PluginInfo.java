/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-8-5
 */
package com.nearme.freeupgrade.lib.db;

import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_APK_PATH;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_DESC;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_FILE_MD5;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_FILE_SIZE;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_FILE_URL;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_ICON_URL;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_NAME;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_PACKAGE_NAME;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_PLUGIN_ID;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_REMOTE_VERSION_CODE;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_REMOTE_VERSION_NAME;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_UPGRADE_COMMENT;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_UPGRADE_TYPE;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_VERSION_CODE;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_VERSION_NAME;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_CURRENT_SIZE;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_STATUS;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_FILE_NAME;

import java.io.Serializable;

import com.nearme.freeupgrade.lib.pinterface.PluginItem;
import com.nearme.freeupgrade.lib.util.Constants;

import android.content.ContentValues;
import android.database.Cursor;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-8-5
 */
public class PluginInfo implements Serializable, Cloneable {
	private static final long serialVersionUID = 6575938948082539848L;
	/**
	 * 升级类型：未安装可下载
	 */
	public static final int UPGRADE_TYPE_DOWNLOAD = 0;
	/**
	 * 升级类型：普通更新
	 */
	public static final int UPGRADE_TYPE_UPGRADE = 1;
	/**
	 * 升级类型：强制更新
	 */
	public static final int UPGRADE_TYPE_FORCE_UPGRADE = 2;

	public String pid;
	public String name;
	public String iconUrl;
	public int versionCode;
	public String versionName;
	public String pkgName;
	public String fileUrl;
	public long fileSize;
	public String fileMd5;
	public int upgradeType;// 升级类型
	public String desc;// 插件描述
	public String upgradeComment;// 最新版插件升级描述
	public int remoteVersionCode;// 最新版本号
	public String remoteVersionName;// 最新版本名
	// 以下与PluginItem不同
	public long currentSize;
	public int status;
	public String apkPath;// 本地文件路径
	public String fileName;
	
	public PluginInfo() {
		remoteVersionCode = -1;
	}

	public Object clone() {
		try {
			return super.clone();
		} catch (Exception e) {
			return null;
		}
	}

	public ContentValues getDatabaseValus() {
		ContentValues values = new ContentValues();
		values.put(COL_PLUGIN_ID, pid);
		values.put(COL_NAME, name);
		values.put(COL_ICON_URL, iconUrl);
		values.put(COL_VERSION_CODE, versionCode);
		values.put(COL_VERSION_NAME, versionName);
		values.put(COL_APK_PATH, apkPath);
		values.put(COL_PACKAGE_NAME, pkgName);
		values.put(COL_FILE_URL, fileUrl);
		values.put(COL_FILE_SIZE, fileSize);
		values.put(COL_FILE_MD5, fileMd5);
		values.put(COL_UPGRADE_TYPE, upgradeType);
		values.put(COL_DESC, desc);
		values.put(COL_UPGRADE_COMMENT, upgradeComment);
		values.put(COL_REMOTE_VERSION_CODE, remoteVersionCode);
		values.put(COL_REMOTE_VERSION_NAME, remoteVersionName);
		values.put(COL_CURRENT_SIZE, currentSize);
		values.put(COL_STATUS, status);
		values.put(COL_FILE_NAME, status);
		return values;
	}

	public void setLocalPluginInfo(Cursor c) {
		pid = c.getString(c.getColumnIndex(COL_PLUGIN_ID));
		name = c.getString(c.getColumnIndex(COL_NAME));
		iconUrl = c.getString(c.getColumnIndex(COL_ICON_URL));
		versionCode = c.getInt(c.getColumnIndex(COL_VERSION_CODE));
		versionName = c.getString(c.getColumnIndex(COL_VERSION_NAME));
		apkPath = c.getString(c.getColumnIndex(COL_APK_PATH));
		pkgName = c.getString(c.getColumnIndex(COL_PACKAGE_NAME));
		fileUrl = c.getString(c.getColumnIndex(COL_FILE_URL));
		fileSize = c.getLong(c.getColumnIndex(COL_FILE_SIZE));
		fileMd5 = c.getString(c.getColumnIndex(COL_FILE_MD5));
		upgradeType = c.getInt(c.getColumnIndex(COL_UPGRADE_TYPE));
		desc = c.getString(c.getColumnIndex(COL_DESC));
		upgradeComment = c.getString(c.getColumnIndex(COL_UPGRADE_COMMENT));
		remoteVersionCode = c.getInt(c.getColumnIndex(COL_REMOTE_VERSION_CODE));
		remoteVersionName = c.getString(c.getColumnIndex(COL_REMOTE_VERSION_NAME));
		currentSize = c.getLong(c.getColumnIndex(COL_CURRENT_SIZE));
		status = c.getInt(c.getColumnIndex(COL_STATUS));
		fileName = c.getString(c.getColumnIndex(COL_FILE_NAME));
	}

	public void setLocalPluginInfo(PluginItem item) {
		pid = item.pid;
		name = item.name;
		iconUrl = item.iconUrl;
		versionCode = item.versionCode;
		versionName = item.versionName;
		pkgName = item.pkgName;
		fileUrl = item.fileUrl;
		if (this.status != Constants.STATUS_DOWNLOADING) {
			fileSize = item.fileSize;
			fileMd5 = item.fileMd5;
		}
		upgradeType = item.upgradeType;
		desc = item.desc;
		upgradeComment = item.upgradeComment;
		remoteVersionCode = item.remoteVersionCode;
		remoteVersionName = item.remoteVersionName;

	}
}
