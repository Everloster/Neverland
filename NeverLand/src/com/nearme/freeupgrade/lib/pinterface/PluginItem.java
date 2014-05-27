/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-8-5
 */
package com.nearme.freeupgrade.lib.pinterface;

import android.os.Parcel;
import android.os.Parcelable;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-8-5
 */
public class PluginItem implements Parcelable {
	/**
	 * 升级类型：未安装可下载
	 */
	public static final int UPGRADE_TYPE_DOWNLOAD = 0;
	/**
	 * 升级类型：已安装无更新
	 */
	public static final int UPGRADE_TYPE_INSTALLED = 1;
	/**
	 * 升级类型：普通更新
	 */
	public static final int UPGRADE_TYPE_UPGRADE = 2;
	/**
	 * 升级类型：强制更新
	 */
	public static final int UPGRADE_TYPE_FORCE_UPGRADE = 3;

	public String pid;
	public String name;
	public String iconUrl;
	public int versionCode;
	public String versionName;
	// public String apkPath;// 本地文件路径
	public String pkgName;
	public String fileUrl;
	public long fileSize;
	public String fileMd5;
	public int upgradeType;// 升级类型
	public String desc;// 插件描述
	public String upgradeComment;// 最新版插件升级描述
	public int remoteVersionCode;// 最新版本号
	public String remoteVersionName;// 最新版本名
	
	public PluginItem() {
		super();
		pid = "";
	}

	public PluginItem(Parcel source) {
		pid = source.readString();
		name = source.readString();
		iconUrl = source.readString();
		versionCode = source.readInt();
		versionName = source.readString();
		// apkPath = source.readString();
		pkgName = source.readString();
		fileUrl = source.readString();
		fileSize = source.readLong();
		fileMd5 = source.readString();
		upgradeType = source.readInt();
		desc = source.readString();
		upgradeComment = source.readString();
		remoteVersionCode = source.readInt();
		remoteVersionName = source.readString();
	}

	@Override
	public int describeContents() {
		return 0;
	}

	@Override
	public void writeToParcel(Parcel dest, int flags) {
		dest.writeString(pid);
		dest.writeString(name);
		dest.writeString(iconUrl);
		dest.writeInt(versionCode);
		dest.writeString(versionName);
		// dest.writeString(apkPath);
		dest.writeString(pkgName);
		dest.writeString(fileUrl);
		dest.writeLong(fileSize);
		dest.writeString(fileMd5);
		dest.writeInt(upgradeType);
		dest.writeString(desc);
		dest.writeString(upgradeComment);
		dest.writeInt(remoteVersionCode);
		dest.writeString(remoteVersionName);
	}

	public static final Parcelable.Creator<PluginItem> CREATOR = new Parcelable.Creator<PluginItem>() {
		public PluginItem createFromParcel(Parcel source) {
			return new PluginItem(source);
		}

		public PluginItem[] newArray(int size) {
			throw new UnsupportedOperationException();
		}
	};
}
