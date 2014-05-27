/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-7-25
 */
package com.nearme.freeupgrade.lib;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.content.pm.PackageInfo;
import android.os.Bundle;
import android.os.IBinder;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.Window;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-7-25
 */
public class PluginActivity extends Activity implements IActivity {
	private Activity mActivity = null;
	protected String mApkPath = "";
	public View mContentView = null;
	private Context mContext = null;
	private ClassLoader mDexClassLoader = null;
	boolean mFinished = false;
	protected boolean mIsRunInPlugin = false;
	protected Activity mOutActivity = null;
	protected PackageInfo mPackageInfo;
	
	protected PluginService mPluginService;
	
	protected ServiceConnection mServiceConnection = new ServiceConnection() {
		public void onServiceConnected(ComponentName name, IBinder service) {
			mPluginService = ((PluginService.MyBinder) service).getService();
		}

		public void onServiceDisconnected(ComponentName name) {

		}
	};

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		if (this.mIsRunInPlugin) {
			this.mActivity = this.mOutActivity;
		} else {
			super.onCreate(savedInstanceState);
			this.mActivity = this;
		}
	}

	@Override
	protected void onDestroy() {
		if (this.mIsRunInPlugin) {
			this.mDexClassLoader = null;
			this.mPackageInfo = null;
			this.mContentView = null;
			this.mActivity = null;
			this.mContext = null;
			this.mOutActivity = null;
		} else {
			super.onDestroy();
		}
	}

	protected void onPause() {
		if (!this.mIsRunInPlugin) {
			super.onPause();
		}
	}

	protected void onRestart() {
		if (!this.mIsRunInPlugin) {
			super.onRestart();
		}
	}

	protected void onResume() {
		if (!this.mIsRunInPlugin) {
			super.onResume();
		}
	}

	protected void onStart() {
		if (!this.mIsRunInPlugin) {
			super.onStart();
		}
	}

	protected void onStop() {
		if (!this.mIsRunInPlugin) {
			super.onStop();
		}
	}

	@Override
	public void setContentView(int paramInt) {
		if (this.mIsRunInPlugin) {
			this.mContentView = LayoutInflater.from(this.mContext).inflate(paramInt, null);
			this.mActivity.setContentView(this.mContentView);
		} else {
			super.setContentView(paramInt);
		}
	}

	public void setContentView(View paramView) {
		this.mContentView = paramView;
		this.mActivity.setContentView(paramView);
	}

	public View findViewById(int paramInt) {
		View localView;
		if ((this.mIsRunInPlugin) && (this.mContentView != null)) {
			localView = this.mContentView.findViewById(paramInt);
		} else {
			localView = super.findViewById(paramInt);
		}
		return localView;
	}

	public void finish() {
		if (this.mIsRunInPlugin) {
			this.mOutActivity.finish();
			this.mFinished = true;
		} else {
			super.finish();
		}
	}

	public LayoutInflater getLayoutInflater() {
		return LayoutInflater.from(this.mContext);
	}

	@Override
	public Intent registerReceiver(BroadcastReceiver receiver, IntentFilter filter) {
		if (this.mIsRunInPlugin) {
			return this.mContext.registerReceiver(receiver, filter);
		} else {
			return super.registerReceiver(receiver, filter);
		}
	}

	@Override
	public void unregisterReceiver(BroadcastReceiver receiver) {
		if (this.mIsRunInPlugin) {
			this.mContext.unregisterReceiver(receiver);
		} else {
			super.unregisterReceiver(receiver);
		}
	}

	@Override
	public Window getWindow() {
		if (this.mIsRunInPlugin) {
			return mOutActivity.getWindow();
		} else {
			return super.getWindow();
		}
	}

	public void startActivity(Intent intent) {
		if (this.mIsRunInPlugin) {
			this.mContext.startActivity(intent);
		} else {
			super.startActivity(intent);
		}
	}
	
	public void startActivityForResult(Intent intent, int requestCode) {
		if (this.mIsRunInPlugin) {
			if (mOutActivity != null) {
				mOutActivity.startActivityForResult(intent, requestCode);
			}
		}
	}
	
	public void startActivityForResult(Intent intent, int requestCode, Bundle bundle) {
		if (this.mIsRunInPlugin) {
			if (mOutActivity != null) {
				mOutActivity.startActivityForResult(intent, requestCode, bundle);
			}
		}
	}

	@Override
	public ComponentName startService(Intent service) {
		if (mIsRunInPlugin) {
			service.setClass(mContext, PluginProxyService.class);
			return mContext.startService(service);
		} else {
			return super.startService(service);
		}
	}

	@Override
	public boolean stopService(Intent name) {
		if (mIsRunInPlugin) {
			name = new Intent(mContext, PluginProxyService.class);
			return mContext.stopService(name);
		} else {
			return super.stopService(name);
		}
	}

	@Override
	public boolean bindService(Intent service, ServiceConnection conn, int flags) {
		if (mIsRunInPlugin) {
			service.setClass(mContext, PluginProxyService.class);
			return mContext.bindService(service, mServiceConnection, flags);
		} else {
			return super.bindService(service, conn, flags);
		}

	}

	@Override
	public void unbindService(ServiceConnection conn) {
		if (mIsRunInPlugin) {
			mContext.unbindService(mServiceConnection);
		} else {
			super.unbindService(conn);
		}
	}

	@Override
	public Context getApplicationContext() {
		if (mIsRunInPlugin) {
			return mContext;
		} else {
			return super.getApplicationContext();
		}
	}

	@Override
	public Context getBaseContext() {
		if (mIsRunInPlugin) {
			return mContext;
		} else {
			return super.getBaseContext();
		}
	}
	
	@Override
	protected void onActivityResult(int requestCode, int resultCode, Intent data) {
		super.onActivityResult(requestCode, resultCode, data);
	}

	/****************************************** 以下是接口实现 ***********************************************/

	@Override
	public void IInit(String paramString, Activity paramActivity, ClassLoader paramClassLoader,
			PackageInfo paramPackageInfo) {
		this.mIsRunInPlugin = true;
		this.mDexClassLoader = paramClassLoader;
		this.mOutActivity = paramActivity;
		this.mApkPath = paramString;
		this.mContext = new PluginContextWrapper(paramActivity, 0, this.mApkPath,
				this.mDexClassLoader);
		this.mPackageInfo = paramPackageInfo;
		attachBaseContext(this.mContext);
	}

	@Override
	public void IOnCreate(Bundle paramBundle) {
		onCreate(paramBundle);
	}

	@Override
	public void IOnDestroy() {
		onDestroy();
	}

	@Override
	public boolean IOnKeyDown(int paramInt, KeyEvent paramKeyEvent) {
		return onKeyDown(paramInt, paramKeyEvent);
	}

	@Override
	public void IOnPause() {
		onPause();
	}

	@Override
	public void IOnResume() {
		onResume();
	}

	@Override
	public void IOnStart() {
		onStart();
	}

	@Override
	public void IOnStop() {
		onStop();
	}

	@Override
	public void ISetIntent(Intent paramIntent) {
		setIntent(paramIntent);
	}

}
