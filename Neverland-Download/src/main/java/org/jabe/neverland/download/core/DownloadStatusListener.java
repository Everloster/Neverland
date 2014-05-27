package org.jabe.neverland.download.core;

public interface DownloadStatusListener extends DownloadListener{
	
	public void onStatusChanged(String packageName, int status);

	public void onFailure(String packageName, Exception e);
	
	public void onStatusCountChanged(int downloadingCount);
	
}
