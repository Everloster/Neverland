package org.jabe.neverland.download.core;

import java.util.ArrayList;
import java.util.List;


public class DownloadListenerManager {
	
	protected DownloadListenerManager() {
	}

	public static final String TAG = DownloadListenerManager.class.getSimpleName();

	protected List<DownloadStatusListener> mDownloadStatusListeners = new ArrayList<DownloadStatusListener>();
	protected List<DownloadProgressListener> mDownloadProgressListeners = new ArrayList<DownloadProgressListener>();

	protected void invokeUpdateProgress(String packageName, double added,
			double downloaded, double total) {
		for (DownloadProgressListener dl : mDownloadProgressListeners) {
			dl.onUpdateProgress(packageName, added, downloaded, total);
		}
	}

	protected void invokeStatusChanged(String packageName, int status) {
		for (DownloadStatusListener dl : mDownloadStatusListeners) {
			dl.onStatusChanged(packageName, status);
		}
	}

	protected void invokeFailure(String packageName, Exception e) {
		for (DownloadStatusListener dl : mDownloadStatusListeners) {
			dl.onFailure(packageName, e);
		}
	}

	public void removeListenter(DownloadListener downloadInterface) {
		if (downloadInterface == null) {
			return;
		}
		
		if (downloadInterface instanceof DownloadProgressListener) {
			mDownloadProgressListeners.remove(downloadInterface);
		} else if (downloadInterface instanceof DownloadStatusListener) {
			mDownloadStatusListeners.remove(downloadInterface);
		}
	}

	public void registerListener(DownloadListener downloadInterface) {
		if (downloadInterface == null) {
			return;
		}
		if (downloadInterface instanceof DownloadProgressListener) {
			if (!mDownloadProgressListeners.contains(downloadInterface)) {
				mDownloadProgressListeners.add((DownloadProgressListener) downloadInterface);
			}
		} else if (downloadInterface instanceof DownloadStatusListener) {
			if (!mDownloadStatusListeners.contains(downloadInterface)) {
				mDownloadStatusListeners.add((DownloadStatusListener) downloadInterface);
			}
		}
	}
	
}
