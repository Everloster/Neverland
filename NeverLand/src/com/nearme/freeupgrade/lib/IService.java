package com.nearme.freeupgrade.lib;

import android.content.Intent;
import android.os.IBinder;

public interface IService {

	abstract void Iinit(String packgeName, String className);

	abstract IBinder IonBind(Intent intent);

	abstract void IonCreate();

	abstract void IonDestroy();

	abstract int IonStartCommand(Intent intent, int flags, int startId);

	abstract boolean IonUnbind(Intent intent);

	abstract void onRebind(Intent intent);
}
