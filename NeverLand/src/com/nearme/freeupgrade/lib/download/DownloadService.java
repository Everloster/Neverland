/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-8-8
 */
package com.nearme.freeupgrade.lib.download;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.ConcurrentModificationException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.IBinder;
import android.os.StatFs;
import android.text.TextUtils;

import com.nearme.freeupgrade.lib.db.DBUtil;
import com.nearme.freeupgrade.lib.db.PluginInfo;
import com.nearme.freeupgrade.lib.pinterface.PluginItem;
import com.nearme.freeupgrade.lib.util.Constants;
import com.nearme.freeupgrade.lib.util.InnerUtil;
import com.nearme.freeupgrade.lib.util.LogUtil;
import com.nearme.freeupgrade.lib.util.NetUtil;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-8-8
 */
public class DownloadService extends Service implements DownloadListener {
	/**
	 * 致命错误
	 */
	public static final int DOWNLOAD_ERROR_CODE_FATAL = 1;
	/**
	 * sd卡不存在
	 */
	public static final int DOWNLOAD_ERROR_CODE_NO_SDCARD = 2;
	/**
	 * sd卡空间不足
	 */
	public static final int DOWNLOAD_ERROR_CODE_NO_SPACE = 3;
	/**
	 * 无网络
	 */
	public static final int DOWNLOAD_ERROR_CODE_NO_NET = 4;
	/**
	 * 未知原因
	 */
	public static final int DOWNLOAD_ERROR_CODE_UNKNOW_REASON = 5;

	/**
	 * 下载、更新状态改变
	 */
	public static final int ACTION_DOWNLOAD_REFRESH = 0;
	public static final int ACTION_DOWNLOAD_CANCEL = 1;
	public static final int ACTION_DOWNLOAD_FAILED = 2;
	public static final int ACTION_DOWNLOAD_START = 4;
	public static final int ACTION_DOWNLOAD_UPDATE = 5;
	public static final int ACTION_DOWNLOAD_STOP = 6;
	public static final int ACTION_DOWNLOAD_SUCCESS = 7;
	public static final int ACTION_UPGRADE_CHANGE = 15;

	/**
	 * 服务开启状态
	 */
	private static final int SERVICE_STATUS_START = 1;
	private static final int SERVICE_STATUS_PAUSE = 2;
	private static final int SERVICE_STATUS_CANCEL = 3;
	private static final int SERVICE_STATUS_PAUSE_ALL = 4;

	private List<DownloadThread> runningTaskList;
	private Map<String, DownloadThread> taskMap;
	private static StatusChangeListener statusChangeListener;
	public static boolean mFromActivity = false;
	private static long lastUpdateTime = 0;// 保存进度更新的最后时间

	@Override
	public void onCreate() {
		LogUtil.i(Constants.TAG, "onCreate()");
		taskMap = new HashMap<String, DownloadThread>();
		runningTaskList = new ArrayList<DownloadThread>();
		// 未正常终止下载的如果服务5秒内启动则自动进行下载
		List<PluginInfo> infos = DBUtil.getDownloadingInfos(this, true);

		for (PluginInfo info : infos) {
			LogUtil.i("Download", "restart name = " + info.name + ":" + mFromActivity);
			if (!mFromActivity) {
				DownloadThread downloadThread = new DownloadThread(getApplicationContext(), info,
						this, -1, false);
				startDownloadThread(downloadThread);
				info.status = Constants.STATUS_WAITING;
				DBUtil.updateDownloadInfoByPkgName(this, info);
				broadcastDownloadStart(info.pkgName);
			} else {
				info.status = Constants.STATUS_PAUSE;
				DBUtil.updateDownloadInfoByPkgName(getApplicationContext(), info);
			}
		}
		mFromActivity = false;
		// refreshInstallMap();
		// refreshUpgradeMap();
		super.onCreate();
	}

	public void broadcastDownloadCancel(String pkgName) {
		broadcastStatusChange(pkgName, ACTION_DOWNLOAD_CANCEL, -1, null);
	}

	public void broadcastDownloadFail(String pkgName, int errorCode, String errorMsg) {
		broadcastStatusChange(pkgName, ACTION_DOWNLOAD_FAILED, -1, null);
	}

	public void broadcastDownloadStart(String pkgName) {
		broadcastStatusChange(pkgName, ACTION_DOWNLOAD_START, -1, null);
	}

	public void broadcastDownloadUpdate(String pkgName) {
		broadcastStatusChange(pkgName, ACTION_DOWNLOAD_UPDATE, -1, null);
	}

	public void broadcastDownloadStop(String pkgName) {
		broadcastStatusChange(pkgName, ACTION_DOWNLOAD_STOP, -1, null);
	}

	public void broadcastDownloadSuccess(String pkgName) {
		broadcastStatusChange(pkgName, ACTION_DOWNLOAD_SUCCESS, -1, null);
	}

	public static void setStatusChangeListener(StatusChangeListener listener) {
		statusChangeListener = listener;
	}

	public static void broadcastStatusChange(String pkgName, int action, int msgCode, String msg) {
		if (statusChangeListener != null) {
			statusChangeListener.onStatusChange(pkgName, action, msgCode, msg);
		}
	}

	@Override
	public void onDestroy() {
		ArrayList<DownloadThread> downloadingThreads = new ArrayList<DownloadThread>();
		downloadingThreads.addAll(runningTaskList);
		downloadingThreads.addAll(taskMap.values());
		for (DownloadThread thread : downloadingThreads) {
			PluginInfo _info = thread.getDownloadInfo();
			stopDownloadThread(_info.pkgName);
			if (_info.status == Constants.STATUS_DOWNLOADING
					|| _info.status == Constants.STATUS_WAITING) {
				_info.status = Constants.STATUS_PAUSE;
			}
			DBUtil.updateDownloadInfoByPkgName(this, _info);
		}
		downloadingThreads.clear();
		LogUtil.i("Download", "service stop");
		super.onDestroy();
	}

	@Override
	public int onStartCommand(Intent intent, int flags, int startId) {
		new CommandThread(this, intent, this).start();
		return super.onStartCommand(intent, flags, startId);
	}

	class CommandThread extends Thread {
		Context context;
		Intent intent;
		DownloadListener downloadListener;

		CommandThread(Context context, Intent intent, DownloadListener downloadListener) {
			this.context = context;
			this.intent = intent;
			this.downloadListener = downloadListener;
		}

		@Override
		public void run() {
			if (intent != null) {
				// String pid =
				// intent.getLongExtra(Constants.EXTRA_KEY_PRODUCT_ID, -1);
				int startStatus = intent.getIntExtra(Constants.EXTRA_KEY_START, -1);
				PluginItem pluginItem = (PluginItem) intent
						.getParcelableExtra(Constants.EXTRA_KEY_PLUGIN_ITEM);
				if (pluginItem != null) {
					PluginInfo bean = DBUtil.getDownloadInfoByPkgName(context, pluginItem.pkgName);
					if (startStatus != SERVICE_STATUS_START
							&& startStatus != SERVICE_STATUS_PAUSE_ALL && bean == null) {
						DBUtil.deleteDownloadInfoByPkgName(getApplicationContext(), pluginItem.pkgName);
						return;
					}
					DownloadThread downloadThread = null;
					boolean needDownloadPreparation = false;
					switch (startStatus) {
						case SERVICE_STATUS_START:
							if (bean == null) {
								bean = new PluginInfo();
								bean.setLocalPluginInfo(pluginItem);
								bean.status = Constants.STATUS_WAITING;
								bean.fileName = bean.pkgName + "." + Constants.SUFFIX_PLUGIN;
								bean.apkPath = InnerUtil.getPluginFile(bean.fileName)
										.getAbsolutePath();
								DBUtil.insertDownloadInfo(context, bean);
							} else {
								bean.status = Constants.STATUS_WAITING;
								DBUtil.updateDownloadInfoByPkgName(context, bean);
							}
							stopDownloadThread(bean.pkgName);
							// TODO DownloadThread第三个参数传哪个versionCode待确定
							downloadThread = new DownloadThread(getApplicationContext(), bean,
									downloadListener, bean.remoteVersionCode,
									needDownloadPreparation);
							startDownloadThread(downloadThread);
							broadcastDownloadStart(bean.pkgName);
							break;
						case SERVICE_STATUS_PAUSE:
							stopDownloadThread(bean.pkgName);
							if (bean.status == Constants.STATUS_DOWNLOADING
									|| bean.status == Constants.STATUS_WAITING) {
								bean.status = Constants.STATUS_PAUSE;
							}
							DBUtil.updateDownloadInfoByPkgName(context, bean);
							broadcastDownloadStop(bean.pkgName);
							break;
						case SERVICE_STATUS_PAUSE_ALL: // 暂停所有的下载线程
							ArrayList<DownloadThread> downloadingThreads = new ArrayList<DownloadThread>();
							downloadingThreads.addAll(runningTaskList);
							downloadingThreads.addAll(taskMap.values());
							for (DownloadThread thread : downloadingThreads) {
								PluginInfo _info = thread.getDownloadInfo();
								stopDownloadThread(_info.pkgName);
								if (_info.status == Constants.STATUS_DOWNLOADING
										|| _info.status == Constants.STATUS_WAITING) {
									_info.status = Constants.STATUS_PAUSE;
								}
								DBUtil.updateDownloadInfoByPkgName(context, _info);
							}
							downloadingThreads.clear();
							break;
						case SERVICE_STATUS_CANCEL:
							stopDownloadThread(bean.pkgName);
							InnerUtil.deleteDownloadTempFile(bean.fileName);
							if (TextUtils.isEmpty(bean.remoteVersionName)) {
								DBUtil.deleteDownloadInfoByPkgName(getApplicationContext(), bean.pkgName);
							} else {
								DBUtil.cancelUpgradeInfoByPkgName(getApplicationContext(), bean.pkgName);
							}
							broadcastDownloadCancel(bean.pkgName);
							break;
					}
				}
			}
			startDownloadThreadIfNeed();
			super.run();
		}
	}

	@Override
	public void onDownloadFail(String pkgName) {
		DownloadThread thread = taskMap.get(pkgName);
		if (thread != null) {
			PluginInfo info = thread.getDownloadInfo();
			stopDownloadThread(pkgName);
			info.status = Constants.STATUS_PAUSE;
			DBUtil.updateDownloadInfoByPkgName(getApplicationContext(), info);
			int errorCode = -1;
			String errorMsg;
			if (!InnerUtil.isSdcardExist()) {
				errorCode = DOWNLOAD_ERROR_CODE_NO_SDCARD;
				errorMsg = "no sdcard";
			} else if (info.fileSize != Long.MAX_VALUE && getLeftSpace() < info.fileSize) {
				errorCode = DOWNLOAD_ERROR_CODE_NO_SPACE;
				errorMsg = "no enough space";
			} else if (!NetUtil.isNetWorking(getApplicationContext())) {
				errorCode = DOWNLOAD_ERROR_CODE_NO_NET;
				errorMsg = "no net";
			} else {
				errorCode = DOWNLOAD_ERROR_CODE_UNKNOW_REASON;
				errorMsg = "unknow reason";
			}
			broadcastDownloadFail(pkgName, errorCode, errorMsg);
		}
		startDownloadThreadIfNeed();
	}

	@Override
	public void onDownloadFatal(String pkgName) {
		DownloadThread thread = taskMap.get(pkgName);
		if (thread != null) {
			PluginInfo info = thread.getDownloadInfo();
			stopDownloadThread(pkgName);
			DBUtil.deleteDownloadInfoByPkgName(getApplicationContext(), info.pkgName);
			InnerUtil.deleteDownloadTempFile(info.fileName);
			broadcastDownloadFail(pkgName, DOWNLOAD_ERROR_CODE_FATAL, "fatal error");
		}
		startDownloadThreadIfNeed();
	}

	@Override
	public void onDownloadStart(String pkgName) {
		DownloadThread thread = taskMap.get(pkgName);
		if (thread != null) {
			PluginInfo info = thread.getDownloadInfo();
			info.status = Constants.STATUS_DOWNLOADING;
			DBUtil.updateDownloadInfoByPkgName(getApplicationContext(), info);
			broadcastDownloadStart(pkgName);
		}
	}

	@Override
	public void onDownloadSuccess(PluginInfo info) {
		info.status = Constants.STATUS_DOWNLOADED;
		stopDownloadThread(info.pkgName);
		DBUtil.updateDownloadInfoByPkgName(getApplicationContext(), info);
		broadcastDownloadSuccess(info.pkgName);
		startDownloadThreadIfNeed();
	}

	@Override
	public void onDownloadUpdate(String pkgName) {
		DownloadThread thread = taskMap.get(pkgName);
		if (thread != null) {
			PluginInfo info = thread.getDownloadInfo();
			DBUtil.updateDownloadInfoByPkgName(getApplicationContext(), info);
			if (System.currentTimeMillis() - lastUpdateTime >= 1000) {
				broadcastDownloadUpdate(info.pkgName);
				lastUpdateTime = System.currentTimeMillis();
			}
		}
	}

	private synchronized void startDownloadThreadIfNeed() {
		int downloadingSize = 0;
		synchronized (taskMap) {
			Collection<DownloadThread> threadCollection = taskMap.values();
			for (DownloadThread thread : threadCollection) {
				if (thread.getDownloadInfo().status == Constants.STATUS_DOWNLOADING
						|| thread.getDownloadInfo().status == Constants.STATUS_WAITING) {
					downloadingSize++;
				}
			}
			while (downloadingSize < 2 && runningTaskList.size() > 0) {
				DownloadThread thread = runningTaskList.remove(0);
				PluginInfo info = thread.getDownloadInfo();
				taskMap.put(info.pkgName, thread);
				thread.start();
				downloadingSize++;
			}
		}
	}

	private void startDownloadThread(DownloadThread thread) {
		runningTaskList.add(thread);
		startDownloadThreadIfNeed();
	}

	private void stopDownloadThread(String pkgName) {
		DownloadThread thread = null;
		synchronized (taskMap) {
			thread = taskMap.remove(pkgName);
		}
		if (thread != null) {
			thread.stopDownload();
		} else {
			try {
				Iterator<DownloadThread> itr = runningTaskList.iterator();
				while (itr.hasNext()) {
					DownloadThread _thread = itr.next();
					if (_thread.getDownloadInfo().pkgName.equals(pkgName)) {
						_thread.stopDownload();
						itr.remove();
					}
				}
			} catch (ConcurrentModificationException e) {
				try {
					Iterator<DownloadThread> itr = runningTaskList.iterator();
					while (itr.hasNext()) {
						DownloadThread _thread = itr.next();
						if (_thread.getDownloadInfo().pkgName.equals(pkgName)) {
							_thread.stopDownload();
							itr.remove();
						}
					}
				} catch (Exception e2) {
				}
			}
		}
		startDownloadThreadIfNeed();
	}

	public static void download(Context ctx, PluginItem item) {
		Intent intent = new Intent(Constants.DOWNLOAD_SERVICE);
		intent.putExtra(Constants.EXTRA_KEY_PLUGIN_ITEM, item);
		intent.putExtra(Constants.EXTRA_KEY_START, SERVICE_STATUS_START);
		ctx.startService(intent);
	}

	public static void resume(Context ctx, PluginItem item) {
		Intent intent = new Intent(Constants.DOWNLOAD_SERVICE);
		intent.putExtra(Constants.EXTRA_KEY_START, SERVICE_STATUS_START);
		intent.putExtra(Constants.EXTRA_KEY_PLUGIN_ITEM, item);
		ctx.startService(intent);
	}

	public static void stopDownload(Context ctx, PluginItem item) {
		Intent intent = new Intent(Constants.DOWNLOAD_SERVICE);
		intent.putExtra(Constants.EXTRA_KEY_PLUGIN_ITEM, item);
		intent.putExtra(Constants.EXTRA_KEY_START, SERVICE_STATUS_PAUSE);
		ctx.startService(intent);
	}

	public static void cancelDownload(Context ctx, PluginItem item) {
		Intent intent = new Intent(Constants.DOWNLOAD_SERVICE);
		intent.putExtra(Constants.EXTRA_KEY_PLUGIN_ITEM, item);
		intent.putExtra(Constants.EXTRA_KEY_START, SERVICE_STATUS_CANCEL);
		ctx.startService(intent);
	}

	public static void pauseAllDownload(Context ctx) {
		Intent intent = new Intent(Constants.DOWNLOAD_SERVICE);
		intent.putExtra(Constants.EXTRA_KEY_START, SERVICE_STATUS_PAUSE_ALL);
		// intent.putExtra(Constants.EXTRA_KEY_PRODUCT_ID, 0l);
		ctx.startService(intent);
	}

	public long getLeftSpace() {
		File file = InnerUtil.getSdRootFile(getApplicationContext());
		StatFs statFs = new StatFs(file.getAbsolutePath());
		return (long) (statFs.getBlockSize() * ((long) statFs.getAvailableBlocks() - 4));
	}

	@Override
	public IBinder onBind(Intent intent) {
		return null;
	}
}
