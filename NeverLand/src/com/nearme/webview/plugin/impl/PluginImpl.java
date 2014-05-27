package com.nearme.webview.plugin.impl;

import com.nearme.webview.plugin.ActivityWrapper;
import com.nearme.webview.plugin.IPlugin;
import com.nearme.webview.plugin.PluginContextWrapper;

import android.app.Activity;
import android.content.pm.PackageInfo;
import android.content.res.AssetManager;
import android.content.res.Resources;
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
		this.mContextProxy = new PluginContextWrapper(parentActivity, 0, apkPath, parentClassLoader, parentPackageInfo.packageName);
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
		if (mRealActivity != null && mContentView != null) {
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
	
	public AssetManager getAssets() {
		if (mContextProxy != null) {
			mContextProxy.getAssets();
		}
		return super.getAssets();
	}

	public ClassLoader getClassLoader() {
		if (mContextProxy != null) {
			mContextProxy.getClassLoader();
		}
		return super.getClassLoader();
	}

	public Resources getResources() {
		if (mContextProxy != null) {
			mContextProxy.getResources();
		}
		return super.getResources();
	}

	public Resources.Theme getTheme() {
		if (mContextProxy != null) {
			mContextProxy.getTheme();
		}
		return super.getTheme();
	}
	
}
