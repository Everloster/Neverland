/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-8-8
 */
package com.nearme.freeupgrade.lib.db;

import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_PACKAGE_NAME;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_STATUS;

import java.util.List;

import android.content.Context;

import com.nearme.freeupgrade.lib.util.Constants;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-8-8
 */
public class DBUtil {
	/**
	 * 查询正在下载的
	 * 
	 * @param context
	 * @param isOrderBy
	 * @return
	 */
	public static List<PluginInfo> getDownloadingInfos(Context context, boolean isOrderBy) {
		String selection = COL_STATUS + "=? or " + COL_STATUS + "=? or " + COL_STATUS + "=?";
		String[] selectionArgs = new String[] { "" + Constants.STATUS_DOWNLOADING,
				"" + Constants.STATUS_WAITING, "" + Constants.STATUS_CONNECTING };
		String orderBy = isOrderBy ? COL_STATUS + " asc" : null;
		return DBBean.getPluginInfos(context, selection, selectionArgs, orderBy);
	}
	
	/**
	 * 插入
	 * 
	 * @param context
	 * @param info
	 */
	public static void insertDownloadInfo(Context context, PluginInfo info) {
		DBBean.insertPluginInfo(context, info);
	}

	/**
	 * 更新
	 * 
	 * @param context
	 * @param info
	 */
	public static void updateDownloadInfoByPkgName(Context context, PluginInfo info) {
		String selection = COL_PACKAGE_NAME + "=?";
		String[] selectionArgs = new String[] { String.valueOf(info.pkgName) };
		DBBean.updatePluginInfo(context, info, selection, selectionArgs);
	}


	/**
	 * 删除
	 * 
	 * @param context
	 * @param pId
	 */
	public static void deleteDownloadInfoByPkgName(Context context, String pkgName) {
		String selection = COL_PACKAGE_NAME + "=?";
		String[] selectionArgs = new String[] { String.valueOf(pkgName) };
		DBBean.deletePluginInfo(context, selection, selectionArgs);
	}

	/**
	 * 查询
	 * 
	 * @param context
	 * @param pId
	 * @return
	 */
	public static PluginInfo getDownloadInfoByPkgName(Context context, String pkgName) {
		String selection = COL_PACKAGE_NAME + "=?";
		String[] selectionArgs = new String[] { String.valueOf(pkgName) };
		return DBBean.getPluginInfo(context, selection, selectionArgs);
	}

	/**
	 * 取消更新
	 * 
	 * @param context
	 * @param pid
	 */
	public static void cancelUpgradeInfoByPkgName(Context context, String pkgName) {
		String selection = COL_PACKAGE_NAME + "=?";
		String[] selectionArgs = new String[] { String.valueOf(pkgName) };
		DBBean.cancelUpgradeDownloadInfo(context, selection, selectionArgs);
	}
}
