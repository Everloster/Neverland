package com.nearme.freeupgrade.lib;

import java.util.List;

import android.app.ActivityManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.ServiceInfo;
import android.os.IBinder;
import android.util.Log;

import com.nearme.freeupgrade.lib.util.Constants;
import com.nearme.freeupgrade.lib.util.InnerUtil;

import dalvik.system.DexClassLoader;

public class PluginProxyService extends Service {

	@Override
	public IBinder onBind(Intent intent) {
		if (intent != null) {
			loadService(intent.getComponent().getClassName(), Constants.PLUGIN_FILE_PATH);
			//getServiceList(this);
		}
		return null;
	}

	@Override
	public void onCreate() {

	}

	
	public void loadService(String className, String filePath) {
		try {
			PackageInfo plocalObject = getPackageManager().getPackageArchiveInfo(filePath,
					PackageManager.GET_SERVICES);
			DexClassLoader localDexClassLoader = InnerUtil.getDexClassLoaderByPath(this, filePath,
					plocalObject.packageName);
			if ((plocalObject.services != null) && (plocalObject.services.length > 0)) {
				for (ServiceInfo service : plocalObject.services) {
					if (className.equals(service.name)) {
						IService pluginService = (IService) localDexClassLoader
								.loadClass(className).newInstance();
						pluginService.IonCreate();
						break;
					}
				}
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	@Override
	public int onStartCommand(Intent intent, int flags, int startId) {
		if (intent != null) {
			loadService(intent.getComponent().getClassName(), Constants.PLUGIN_FILE_PATH);
			//getServiceList(this);
		}
		return super.onStartCommand(intent, flags, startId);
	}

	public void getServiceList(Context context) {
		ActivityManager activityManager = (ActivityManager) context
				.getSystemService(Context.ACTIVITY_SERVICE);
		List<ActivityManager.RunningServiceInfo> serviceList = activityManager
				.getRunningServices(30);
		if (!(serviceList.size() > 0)) {
			return;
		}
		for (int i = 0; i < serviceList.size(); i++) {

			Log.e("AAAA", "name = " + serviceList.get(i).service.getClassName());
		}
	}

	@Override
	public boolean onUnbind(Intent intent) {
		//getServiceList(this);
		return super.onUnbind(intent);
	}

}
