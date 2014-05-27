package org.jabe.neverland.download.core;

import java.io.File;

public class DownloadManager extends DownloadEngineWraper {
	
	private DownloadConfig mDownloadConfig;

	public DownloadManager(DownloadConfig downloadConfig) {
		super(downloadConfig.mDownloadEngine, downloadConfig.mMessageDeliver);
		this.mDownloadConfig = downloadConfig;
	}
	
	public void downloadOperation(DownloadInfo downloadInfo, DownloadStatus lastStatus) {
		mDownloadConfig.mDownloadOperationMaper.operationMap(downloadInfo, lastStatus);
	}
	
	public File getDownloadedFile(DownloadInfo downloadInfo) {
		final String path = mDownloadConfig.mProgressCacheManager.generateFinishPath(downloadInfo);
		return new File(path);
	}

}
