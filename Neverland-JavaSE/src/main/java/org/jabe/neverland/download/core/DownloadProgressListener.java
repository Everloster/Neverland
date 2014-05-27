package org.jabe.neverland.download.core;

public interface DownloadProgressListener extends DownloadListener {
	
	public void onUpdateProgress(String packageName, double added,
			double downloaded, double total);
	
}
