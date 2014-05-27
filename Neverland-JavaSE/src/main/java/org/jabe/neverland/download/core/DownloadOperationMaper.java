package org.jabe.neverland.download.core;

public abstract class DownloadOperationMaper {
	
	protected DownloadEngine mDownloadEngine;
	
	protected DownloadOperationMaper(DownloadEngine downloadEngine) {
		this.mDownloadEngine = downloadEngine;
	}
	
	public abstract void operationMap(DownloadInfo downloadInfo, DownloadStatus lastStatus);
	
}
