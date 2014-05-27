package org.jabe.neverland.plugin;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.view.KeyEvent;


public interface IPlugin {

	public void onPluginCreate(String apkPath, Activity parentActivity,
			ClassLoader parentClassLoader, PackageInfo parentPackageInfo);

	public void onPluginDestroy();

	public boolean onKeyDown(int paramInt, KeyEvent paramKeyEvent);

	public void onPluginPause();

	public void onPluginResume();

	public void onPluginStart();

	public void onPluginStop();
	
	public void onPluginActivityResult(int requestCode, int resultCode, Intent data);
	
}
