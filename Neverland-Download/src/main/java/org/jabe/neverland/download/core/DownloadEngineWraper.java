package org.jabe.neverland.download.core;


public class DownloadEngineWraper implements DownloadEngine, DownloadRegister{
	
	protected DownloadEngine mDownloadEngine;
	
	protected DownloadListenerWraper mDownloadListenerWraper;
	
	protected DownloadEngineWraper(DownloadEngine mDownloadEngine, AbstractMessageDeliver deliver) {
		super();
		this.mDownloadEngine = mDownloadEngine;
		this.mDownloadListenerWraper = new DownloadListenerWraper(deliver);
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

	@Override
	public void registerListener(DownloadListener downloadInterface) {
		mDownloadListenerWraper.registerListener(downloadInterface);
	}

	@Override
	public void removeListenter(DownloadListener downloadInterface) {
		mDownloadListenerWraper.removeListenter(downloadInterface);
	}

	/* (non-Javadoc)
	 * @see org.jabe.neverland.download.core.DownloadEngine#setIoDownloader(org.jabe.neverland.download.core.engine.impl.IODownloader)
	 */
	@Override
	public void setIoDownloader(IODownloader downloader) {
		mDownloadEngine.setIoDownloader(downloader);
	}
}
