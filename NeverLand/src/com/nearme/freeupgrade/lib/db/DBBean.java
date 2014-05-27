/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-8-8
 */
package com.nearme.freeupgrade.lib.db;

import static com.nearme.freeupgrade.lib.db.PluginProvider.TABLE_PLUGIN;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_CURRENT_SIZE;
import static com.nearme.freeupgrade.lib.db.PluginProvider.COL_STATUS;

import java.util.ArrayList;
import java.util.List;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;

import com.nearme.freeupgrade.lib.util.Constants;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-8-8
 */
public class DBBean {

	/**
	 * 返回插件表Uri
	 * 
	 * @param context
	 * @return
	 */
	private static Uri getPluginUri(Context context) {
		return Uri.parse("content://" + context.getPackageName() + "/" + TABLE_PLUGIN);
	}

	/**
	 * 查询多PluginInfo
	 * 
	 * @param context
	 * @param selection
	 * @param selectionArgs
	 * @param orderBy
	 * @return
	 */
	public static List<PluginInfo> getPluginInfos(Context context, String selection,
			String[] selectionArgs, String orderBy) {
		List<PluginInfo> infos = new ArrayList<PluginInfo>();
		PluginInfo info = null;
		final ContentResolver cr = context.getContentResolver();
		Cursor c = cr.query(getPluginUri(context), null, selection, selectionArgs, orderBy);
		if (c != null) {
			if (c.getCount() > 0 && c.moveToFirst()) {
				do {
					info = new PluginInfo();
					info.setLocalPluginInfo(c);
					infos.add(info);
				} while (c.moveToNext());
			}
			c.close();
		}
		return infos;
	}

	/**
	 * 查询PluginInfo
	 * 
	 * @param context
	 * @param selection
	 * @param selectionArgs
	 * @return
	 */
	public static PluginInfo getPluginInfo(Context context, String selection, String[] selectionArgs) {
		PluginInfo item = null;
		final ContentResolver cr = context.getContentResolver();
		Uri uri = getPluginUri(context);
		Cursor c = cr.query(uri, null, selection, selectionArgs, null);
		if (c != null) {
			if (c.getCount() > 0 && c.moveToFirst()) {
				do {
					item = new PluginInfo();
					item.setLocalPluginInfo(c);
				} while (c.moveToNext());
			}
			c.close();
		}
		return item;
	}

	/**
	 * 删除PluginInfo
	 * 
	 * @param context
	 * @param selection
	 * @param selectionArgs
	 * @return
	 */
	public static int deletePluginInfo(Context context, String selection, String[] selectionArgs) {
		final ContentResolver cr = context.getContentResolver();
		int result = cr.delete(getPluginUri(context), selection, selectionArgs);
		return result;
	}

	/**
	 * 更新PluginInfo
	 * 
	 * @param context
	 * @param info
	 * @param selection
	 * @param selectionArgs
	 */
	public static int updatePluginInfo(Context context, final PluginInfo info, String selection,
			String[] selectionArgs) {
		final ContentResolver cr = context.getContentResolver();
		int result = cr.update(getPluginUri(context), info.getDatabaseValus(), selection,
				selectionArgs);
		return result;
	}

	/**
	 * 初始化插入PluginInfo
	 * 
	 * @param context
	 * @param info
	 */
	public static void insertPluginInfo(Context context, PluginInfo info) {
		final ContentResolver cr = context.getContentResolver();
		cr.insert(getPluginUri(context), info.getDatabaseValus());
	}

	/**
	 * 取消更新
	 * 
	 * @param context
	 * @param selection
	 * @param selectionArgs
	 * @return
	 */
	public static int cancelUpgradeDownloadInfo(Context context, String selection,
			String[] selectionArgs) {
		final ContentResolver cr = context.getContentResolver();
		ContentValues values = new ContentValues();
		values.put(COL_STATUS, Constants.STATUS_DOWNLOADED);
		values.put(COL_CURRENT_SIZE, 0);
		int result = cr.update(getPluginUri(context), values, selection, selectionArgs);
		return result;
	}

}
