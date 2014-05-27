package com.nearme.freeupgrade.lib;

import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.IBinder;

public class PluginService extends Service implements IService {

	private final MyBinder mBinder = new MyBinder();

	public class MyBinder extends Binder {
		PluginService getService() {
			return PluginService.this;
		}
	}

	@Override
	public void Iinit(String packgeName, String className) {
	}

	@Override
	public IBinder IonBind(Intent intent) {
		return onBind(intent);
	}

	@Override
	public void IonCreate() {
		onCreate();
	}

	@Override
	public void IonDestroy() {
		onDestroy();
	}

	@Override
	public int IonStartCommand(Intent intent, int flags, int startId) {
		return onStartCommand(intent, flags, startId);
	}

	@Override
	public boolean IonUnbind(Intent intent) {
		return onUnbind(intent);
	}

	@Override
	public IBinder onBind(Intent intent) {
		return mBinder;
	}
}
