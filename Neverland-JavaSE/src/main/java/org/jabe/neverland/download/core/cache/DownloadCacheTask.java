package org.jabe.neverland.download.core.cache;

import java.io.IOException;

import org.jabe.neverland.download.core.DownloadInfo;

/**
 * 
 * A task use to udpate downloaded-cache in manager
 * 
 * @author Jabe 2013-8-1
 * 
 */
public class DownloadCacheTask {
	
	public long mDownloadedLength = 0;
	
	public long[] mSectionsOffset;
	public int mWorkerCount;
	public int mSectionCount;
	
	public long mContentLength;
	public DownloadInfo mDownloadInfo;
	
	private DownloadCacheManager mProgressCacheManager;
	
	public DownloadCacheTask(DownloadCacheManager progressCacheManager, DownloadInfo downloadInfo) {
		this.mProgressCacheManager = progressCacheManager;
		this.mDownloadInfo = downloadInfo;
		init();
	}
	
	private void init() {
		mSectionCount = mDownloadInfo.getInt(DownloadInfo.P_SECTION_COUNT, 1);
		mWorkerCount = mDownloadInfo.getInt(DownloadInfo.P_WORKER_COUNT, 1);
		initSectionOffset();
	}

	private void initSectionOffset() {
		mSectionsOffset = new long[mSectionCount];
		for (int i = 0; i < mSectionCount; i++) {
			mSectionsOffset[i] = 0;
		}
	}
	
	public void readFromCache() throws IOException {
		mProgressCacheManager.readFromCache(this);
	}
	
	public void saveToCache() throws IOException {
		mProgressCacheManager.saveToCache(this);
	}
	
	public String generateCacheSaveFullPath() {
		return mProgressCacheManager.generateCacheSaveFullPath(mDownloadInfo);
	}
	
	public boolean isInCache() {
		return mProgressCacheManager.isInCache(this);
	}
	
	public boolean checkFinish() {
		return mProgressCacheManager.completeCacheTask(this);
	}
	
	public void updateSectionProgress(byte[] bytes, int sectionNo, long progress) throws IOException {
		mProgressCacheManager.updateSectionProgress(bytes, sectionNo, progress, this);
	}

}
