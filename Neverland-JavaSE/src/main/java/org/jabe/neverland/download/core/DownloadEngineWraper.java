package org.jabe.neverland.download.core;

public class DownloadEngineWraper extends DownloadListenerManager implements DownloadEngine {
	
	protected DownloadEngine mDownloadEngine;
	
	protected DownloadEngineWraper(DownloadEngine mDownloadEngine) {
		super();
		this.mDownloadEngine = mDownloadEngine;
	}

	@Override
	public boolean startDownload(DownloadInfo downloadInfo) {
		return mDownloadEngine.startDownload(downloadInfo);
	}

	@Override
	public boolean resumeDownload(DownloadInfo downloadInfo) {
		return mDownloadEngine.resumeDownload(downloadInfo);
	}

	@Override
	public boolean pauseDownload(DownloadInfo downloadInfo) {
		return mDownloadEngine.pauseDownload(downloadInfo);
	}

	@Override
	public boolean cancelDownload(DownloadInfo downloadInfo) {
		return mDownloadEngine.cancelDownload(downloadInfo);
	}

	@Override
	public boolean restartDownload(DownloadInfo downloadInfo) {
		return mDownloadEngine.restartDownload(downloadInfo);
	}
}
