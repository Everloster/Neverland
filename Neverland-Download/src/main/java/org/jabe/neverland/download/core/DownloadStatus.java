package org.jabe.neverland.download.core;

public enum DownloadStatus {
	DOWNLOAD_STATUS_STARTED, DOWNLOAD_STATUS_RESUME, 
	DOWNLOAD_STATUS_PAUSED, DOWNLOAD_STATUS_FINISHED, 
	DOWNLOAD_STATUS_FAILED, DOWNLOAD_STATUS_UPDATE,
	DOWNLOAD_STATUS_PREPARE, DOWNLOAD_STATUS_INSTALLED,
	DOWNLOAD_STATUS_CANCEL, DOWNLOAD_STATUS_UNINSTALL,
	DOWNLOAD_STATUS_INSTALLING, DOWNLOAD_STATUS_WAIT
}
