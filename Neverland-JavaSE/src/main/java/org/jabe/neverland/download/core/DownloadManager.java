package org.jabe.neverland.download.core;

public class DownloadManager extends DownloadEngineWraper {
	
	private DownloadConfig mDownloadConfig;

	public DownloadManager(DownloadConfig downloadConfig) {
		super(downloadConfig.mDownloadEngine);
		this.mDownloadConfig = downloadConfig;
	}
	
	public void downloadOperation(DownloadInfo downloadInfo, DownloadStatus lastStatus) {
		mDownloadConfig.mDownloadOperationMaper.operationMap(downloadInfo, lastStatus);
	}

}
