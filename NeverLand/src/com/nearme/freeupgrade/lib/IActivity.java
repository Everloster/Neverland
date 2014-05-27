/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-7-25
 */
package com.nearme.freeupgrade.lib;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.os.Bundle;
import android.view.KeyEvent;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-7-25
 */
public interface IActivity {

	public abstract void IInit(String apkPath, Activity paramActivity,
			ClassLoader paramClassLoader, PackageInfo paramPackageInfo);

	public abstract void IOnCreate(Bundle paramBundle);

	public abstract void IOnDestroy();

	public abstract boolean IOnKeyDown(int paramInt, KeyEvent paramKeyEvent);

	public abstract void IOnPause();

	public abstract void IOnResume();

	public abstract void IOnStart();

	public abstract void IOnStop();

	public abstract void ISetIntent(Intent paramIntent);
	
}
