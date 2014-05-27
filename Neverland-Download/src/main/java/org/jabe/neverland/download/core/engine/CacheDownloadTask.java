/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年3月6日
 */
package org.jabe.neverland.download.core.engine;

import org.jabe.neverland.download.core.CacheDownloadInfo;

/**
 * 
 * @Author	LaiLong
 * @Since	2014年3月6日
 */
public abstract class CacheDownloadTask extends DownloadTask {
	
	public void clear() {
		mCacheDownloadInfo = null;
		mCacheInvoker = null;
	}

	/**
	 * 
	 */
	public CacheDownloadTask(DownloadCacheInvoker cacheInvoker) {
		super();
		mCacheDownloadInfo = cacheInvoker.mCacheDownloadInfo;
		mCacheInvoker = cacheInvoker;
	}
	
	protected String getPackageName() {
		return mCacheDownloadInfo.mDownloadInfo.getmPackageName();
	}
	
	protected volatile CacheDownloadInfo mCacheDownloadInfo;
	protected DownloadCacheInvoker mCacheInvoker;
	
	
}
