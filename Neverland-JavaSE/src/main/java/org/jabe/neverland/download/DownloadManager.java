package org.jabe.neverland.download;

import java.util.ArrayList;
import java.util.List;

public abstract class DownloadManager {
	public enum DownloadStatus {
		DOWNLOAD_STATUS_STARTED, DOWNLOAD_STATUS_RESUME, 
		DOWNLOAD_STATUS_PAUSED, DOWNLOAD_STATUS_FINISHED, 
		DOWNLOAD_STATUS_FAILED
	}

	public static final String TAG = DownloadManager.class.getSimpleName();

	
	protected List<DownloadInterface> mDownloadListeners = new ArrayList<DownloadInterface>();
	
	public void registerListener(DownloadInterface downloadInterface) {
		mDownloadListeners.add(downloadInterface);
	}
	
	public void removeListenter(DownloadInterface downloadInterface) {
		mDownloadListeners.remove(downloadInterface);
	}
	
	public abstract void startDownload(String url, String tag);
	
	public abstract boolean resumeDownload(String tag);
	
	public abstract boolean pauseDownload(String tag);
	
	public abstract boolean cancelDownload(String tag);
	
	public abstract boolean  restartDownload(String tag);
	
	public abstract boolean hasDownloadingFile(String url, String tag);
	
	public abstract boolean isDownloading(String tag);
	
	public abstract boolean hasFinished(String url, String tag);
	
	public abstract String generateFileName(String url, String tag);
	
}
