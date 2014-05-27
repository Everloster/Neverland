/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-8-7
 */
package com.nearme.freeupgrade.lib.db;

import android.content.ContentProvider;
import android.content.ContentUris;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.database.sqlite.SQLiteQueryBuilder;
import android.net.Uri;
import android.text.TextUtils;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-8-7
 */
public class PluginProvider extends ContentProvider {
	private SQLiteOpenHelper mOpenHelper;
	//数据库信息
	private static final String DB_NAME = "plugin.db";
	private static final int DB_VERSION = 1;
	private static final String PARAMETER_NOTIFY = "notify";
	//表明
	public static final String TABLE_PLUGIN = "plugin";
	//列明
	public static final String COL_ID = "_id";
	public static final String COL_PLUGIN_ID = "id";
	public static final String COL_NAME = "name";
	public static final String COL_ICON_URL = "iconUrl";
	public static final String COL_VERSION_CODE = "versionCode";
	public static final String COL_VERSION_NAME = "versionName";
	public static final String COL_APK_PATH = "apkPath";// 文件路径
	public static final String COL_PACKAGE_NAME = "pkgName";
	public static final String COL_FILE_URL = "fileUrl";
	public static final String COL_FILE_SIZE = "fileSize";
	public static final String COL_FILE_MD5 = "fileMd5";
	public static final String COL_FILE_NAME = "fileName";
	public static final String COL_UPGRADE_TYPE = "upgradeType";
	public static final String COL_DESC = "desc";
	public static final String COL_UPGRADE_COMMENT = "upgradeComment";
	public static final String COL_REMOTE_VERSION_CODE = "remoteVersionCode";
	public static final String COL_REMOTE_VERSION_NAME = "remoteVersionName";
	public static final String COL_CURRENT_SIZE = "currentSize";// 已下载大小
	public static final String COL_STATUS = "status";// 下载状态
	
	
	private static class DatabaseHelper extends SQLiteOpenHelper {

		public DatabaseHelper(Context context) {
			super(context, DB_NAME, null, DB_VERSION);
		}

		@Override
		public void onCreate(SQLiteDatabase db) {
			db.execSQL("CREATE TABLE " + TABLE_PLUGIN + " (" 
					+ COL_ID + " INTEGER PRIMARY KEY AUTOINCREMENT ,"
					+ COL_PLUGIN_ID + " TEXT ," 
					+ COL_NAME + " TEXT,"
					+ COL_ICON_URL + " TEXT,"
					+ COL_VERSION_CODE + " INTEGER,"
					+ COL_VERSION_NAME + " TEXT,"
					+ COL_APK_PATH + " TEXT,"
					+ COL_PACKAGE_NAME + " TEXT,"
					+ COL_FILE_URL + " TEXT,"
					+ COL_FILE_NAME + " TEXT,"
					+ COL_FILE_SIZE + " INTEGER,"
					+ COL_CURRENT_SIZE + " INTEGER,"
					+ COL_STATUS + " INTEGER,"
					+ COL_FILE_MD5 + " TEXT,"
					+ COL_UPGRADE_TYPE + " INTEGER,"
					+ COL_DESC + " TEXT,"
					+ COL_UPGRADE_COMMENT + " TEXT,"
					+ COL_REMOTE_VERSION_CODE + " INTEGER,"
					+ COL_REMOTE_VERSION_NAME + " TEXT);");
		}

		@Override
		public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
			// TODO Auto-generated method stub

		}

	}

	// 以下不动
	
	@Override
	public boolean onCreate() {
		mOpenHelper = new DatabaseHelper(getContext());
		return true;
	}

	@Override
	public int delete(Uri uri, String selection, String[] selectionArgs) {
		SqlArguments args = new SqlArguments(uri, selection, selectionArgs);

		SQLiteDatabase db = mOpenHelper.getWritableDatabase();
		int count = db.delete(args.table, args.where, args.args);
		if (count > 0)
			sendNotify(uri);
		return count;
	}

	@Override
	public String getType(Uri uri) {
		SqlArguments args = new SqlArguments(uri, null, null);
		if (TextUtils.isEmpty(args.where)) {
			return "vnd.android.cursor.dir/" + args.table;
		} else {
			return "vnd.android.cursor.item/" + args.table;
		}
	}

	private void sendNotify(Uri uri) {
		String notify = uri.getQueryParameter(PARAMETER_NOTIFY);
		if (notify == null || "true".equals(notify)) {
			getContext().getContentResolver().notifyChange(uri, null);
		}
	}

	@Override
	public int bulkInsert(Uri uri, ContentValues[] values) {
		SqlArguments args = new SqlArguments(uri);

		SQLiteDatabase db = mOpenHelper.getWritableDatabase();
		db.beginTransaction();
		try {
			int numValues = values.length;
			for (int i = 0; i < numValues; i++) {
				if (db.insert(args.table, null, values[i]) < 0)
					return 0;
			}
			db.setTransactionSuccessful();
		} finally {
			db.endTransaction();
		}

		sendNotify(uri);
		return values.length;
	}

	@Override
	public Uri insert(Uri uri, ContentValues values) {
		SqlArguments args = new SqlArguments(uri);
		SQLiteDatabase db = mOpenHelper.getWritableDatabase();
		final long rowId = db.insert(args.table, null, values);
		if (rowId <= 0)
			return null;

		uri = ContentUris.withAppendedId(uri, rowId);
		sendNotify(uri);

		return uri;
	}

	@Override
	public Cursor query(Uri uri, String[] projection, String selection, String[] selectionArgs,
			String sortOrder) {
		SqlArguments args = new SqlArguments(uri, selection, selectionArgs);
		SQLiteQueryBuilder qb = new SQLiteQueryBuilder();
		qb.setTables(args.table);

		SQLiteDatabase db = mOpenHelper.getReadableDatabase();
		Cursor result = qb.query(db, projection, args.where, args.args, null, null, sortOrder);
		result.setNotificationUri(getContext().getContentResolver(), uri);

		return result;
	}

	@Override
	public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
		SqlArguments args = new SqlArguments(uri, selection, selectionArgs);

		SQLiteDatabase db = mOpenHelper.getWritableDatabase();
		int count = db.update(args.table, values, args.where, args.args);
		if (count > 0)
			sendNotify(uri);

		return count;
	}

	private static final class SqlArguments {
		public final String[] args;
		public final String table;
		public final String where;

		public SqlArguments(Uri url) {
			if (url.getPathSegments().size() == 1) {
				table = url.getPathSegments().get(0);
				where = null;
				args = null;
			} else {
				throw new IllegalArgumentException("Invalid URI: " + url);
			}
		}

		public SqlArguments(Uri url, String where, String[] args) {
			if (url.getPathSegments().size() == 1) {
				this.table = url.getPathSegments().get(0);
				this.where = where;
				this.args = args;
			} else if (url.getPathSegments().size() != 2) {
				throw new IllegalArgumentException("Invalid URI: " + url);
			} else if (!TextUtils.isEmpty(where)) {
				throw new UnsupportedOperationException("WHERE clause not supported: " + url);
			} else {
				this.table = url.getPathSegments().get(0);
				this.where = "_id=" + ContentUris.parseId(url);
				this.args = null;
			}
		}
	}
}
