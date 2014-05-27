package org.jabe.neverland.download;

public interface DownloadInterface {
	
	public void onStatusChanged(String packageName, int status);

	public void onUpdateProgress(String packageName, double added, double downloaded, double total);

	public void onFailure(String packageName, Exception e);
	
}
