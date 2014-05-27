package org.jabe.neverland.download.core.engine.impl;

import org.jabe.neverland.download.core.DownloadEngine;
import org.jabe.neverland.download.core.DownloadInfo;
import org.jabe.neverland.download.core.DownloadOperationMaper;
import org.jabe.neverland.download.core.DownloadStatus;

public class DefaultOperationMaper extends DownloadOperationMaper {

	public DefaultOperationMaper(DownloadEngine downloadEngine) {
		super(downloadEngine);
	}

	@Override
	public void operationMap(DownloadInfo downloadInfo,
			DownloadStatus lastStatus) {
		switch (lastStatus) {
		case DOWNLOAD_STATUS_FAILED:
		case DOWNLOAD_STATUS_CANCEL:
		case DOWNLOAD_STATUS_UNINSTALL:
		case DOWNLOAD_STATUS_UPDATE:
			mDownloadEngine.startDownload(downloadInfo);
			break;
		case DOWNLOAD_STATUS_PAUSED:
			mDownloadEngine.restartDownload(downloadInfo);
			break;
		case DOWNLOAD_STATUS_PREPARE:
			// prepare do nothing~
			break;
		case DOWNLOAD_STATUS_STARTED:
		case DOWNLOAD_STATUS_RESUME:
			mDownloadEngine.pauseDownload(downloadInfo);
			break;
		case DOWNLOAD_STATUS_FINISHED:
			//TODO show  install
			break;
		case DOWNLOAD_STATUS_INSTALLED:
			//TODO open it
			break;
		default:
			break;
		}
	}
}
