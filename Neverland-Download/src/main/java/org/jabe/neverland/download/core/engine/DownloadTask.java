/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年3月6日
 */
package org.jabe.neverland.download.core.engine;


/**
 * 
 * @Author	LaiLong
 * @Since	2014年3月6日
 */
public abstract class DownloadTask implements Runnable{

	protected volatile boolean keepRunning = false;
	
	/**
	 * 
	 */
	public DownloadTask() {
	}
	
	public abstract boolean start();
	
	public abstract boolean resume();

	public abstract boolean cancel();
	
	public abstract boolean pause();
	
	public abstract void clearCache();
	
	public abstract void stop();
	
	public abstract boolean isDownloading();
	
}
