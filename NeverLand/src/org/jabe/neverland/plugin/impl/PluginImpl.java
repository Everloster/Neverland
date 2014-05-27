package org.jabe.neverland.plugin.impl;

import org.jabe.neverland.plugin.ActivityWrapper;
import org.jabe.neverland.plugin.IPlugin;
import org.jabe.neverland.plugin.PluginContextWrapper;

import android.app.Activity;
import android.content.pm.PackageInfo;
import android.view.LayoutInflater;
import android.view.View;

public abstract class PluginImpl extends ActivityWrapper implements IPlugin {

	protected PluginContextWrapper mContextProxy;
	
	protected Activity mRealActivity;
	
	protected String mApkPath;
	
	protected PackageInfo mParentPackageInfo;
	
	private View mContentView = null;
	

	@Override
	public void onPluginCreate(String apkPath, Activity parentActivity,
			ClassLoader parentClassLoader, PackageInfo parentPackageInfo) {
		this.mApkPath = apkPath;
		this.mParentPackageInfo = parentPackageInfo;
		this.mRealActivity = parentActivity;
		this.mContextProxy = new PluginContextWrapper(parentActivity, 0, apkPath, parentClassLoader);
		attachBaseActivity(parentActivity);
		attachBaseContext(mContextProxy);
		onCreate(null);
	}
	
	@Override
	public View findViewById(int id) {
		if (mContentView != null) {
			return mContentView.findViewById(id);
		} else {
			return null;
		}
	}
	
	@Override
	public void setContentView(int layoutResID) {
		mContentView = LayoutInflater.from(mContextProxy).inflate(layoutResID, null);
		if (mRealActivity != null) {
			mRealActivity.setContentView(mContentView);
		}
	}
	
	@Override
	public void setContentView(View view) {
		mContentView = view;
		if (mRealActivity != null) {
			mRealActivity.setContentView(mContentView);
		}
	}
	
	@Override
	public LayoutInflater getLayoutInflater() {
		return LayoutInflater.from(mContextProxy);
	}
	
}
