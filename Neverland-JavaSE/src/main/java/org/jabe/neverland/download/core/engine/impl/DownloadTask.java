package org.jabe.neverland.download.core.engine.impl;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import org.jabe.neverland.download.core.cache.DownloadCacheTask;
import org.jabe.neverland.download.core.engine.impl.IODownloader.SizeBean;
import org.jabe.neverland.download.util.IoUtils;


public class DownloadTask implements Runnable {
	
	private static final int BUFFER_SIZE = 16 * 1024;
	
	protected TaskConfig mTaskConfig;
	
	public DownloadTask(TaskConfig mTaskConfig) {
		super();
		this.mTaskConfig = mTaskConfig;
		mCacheTask = mTaskConfig.mCacheTask;
	}

	private Boolean[] successTag;
	private DownloadCacheTask mCacheTask;
	private volatile boolean hasStarted = false;
	
	@Override
	public void run() {
		hasStarted = true;
		// task life cycle
		mTaskConfig.mDownloadTaskListener.onPreTask();
		
		try {
			// whatever, need to get content length to check the task in cache
			mCacheTask.mContentLength = getContentLength(mCacheTask.mDownloadInfo.getmDownloadUrl());
			if (mCacheTask.isInCache()) {
				mCacheTask.readFromCache();
			} else {
				mCacheTask.saveToCache();
			}
		} catch (Exception e) {
			
		}
		
		final int secCount = mCacheTask.mSectionCount;

		// task life cycle
		mTaskConfig.mDownloadTaskListener.onBeforeExecute();
		
		if (mCacheTask.mDownloadedLength == mCacheTask.mContentLength && mCacheTask.mContentLength > 0) {
			onSuccess();
			return;
		}
		
		successTag = new Boolean[secCount];
		for (int i = 0; i < successTag.length; i++) {
			successTag[i] = false;
		}
		try {
			if (secCount > 1) {
				doMultipleWork();
			} else {
				doSingleWork();
			}
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}
	
	private void doSingleWork() throws IOException {
		doRealDownload(1, mCacheTask.mContentLength, mCacheTask.mDownloadedLength, mCacheTask.mContentLength);
	}
	
	private void doMultipleWork() throws FileNotFoundException {
		final int secCount = mCacheTask.mSectionCount;
//		mCacheAccessFile = new RandomAccessFile(mCacheTask.generateCacheSaveFullPath(), "rw");
		for (int j = 0; j < secCount; j++) {
			final long length = mCacheTask.mContentLength;
			final long start = mCacheTask.mSectionsOffset[j];
			final long per = length / secCount;
			long endt = -1;
			// 计算当前块的长度
			if (j < secCount - 1) {// 当前块是否为最后一块
				endt = per * (j + 1);
			} else {
				endt = length;
			}
			final long endF = endt;
			final int s = j + 1;
			final long realStart = j * per + start;
			mTaskConfig.mDownloadExecutorService.execute(new Runnable() {
				
				@Override
				public void run() {
					try {
						doRealDownload(s, length, realStart, endF);
					} catch (IOException e) {
						// TODO
						triggerSectionFailure(s);
					}
				}
			});
		}
	}

	private void doRealDownload(final int sectionNo, final long contentLength, final long start, final long end) throws IOException {
		final SizeBean sb = new SizeBean(contentLength, start, end);
		final InputStream is = mTaskConfig.mDownloader.getStream(mCacheTask.mDownloadInfo.getmDownloadUrl(), null, sb);
		try {
			try {
				byte[] bytes = new byte[BUFFER_SIZE];
				while (true) {
					int count = is.read(bytes, 0, BUFFER_SIZE);
					if (count == -1) {
						break;
					}
//					mCacheAccessFile.seek((sectionNo - 1) * per + mCacheTask.mSectionsOffset[sectionNo - 1]);
//					mCacheAccessFile.write(bytes, 0, count);
					mCacheTask.updateSectionProgress(bytes, sectionNo, count);
					mTaskConfig.mDownloadTaskListener.onUpdateProgress(count, mCacheTask.mDownloadedLength, mCacheTask.mContentLength);
				}
				triggerSuccess(sectionNo);
			} finally {
			}
		} finally {
			IoUtils.closeSilently(is);
		}
	}
	
	private long getContentLength(String url) throws IOException {
		URL u = new URL(url);
		HttpURLConnection conn = (HttpURLConnection) u.openConnection();
		long l = conn.getContentLength();
		conn.disconnect();
		return l;
	}
	
	private void triggerSectionFailure(int section) {
		
	}
	
	private void triggerSuccess(int sectionNo) {
		successTag[sectionNo - 1] = true;
		checkAllSuccess();
	}

	private void checkAllSuccess() {
		boolean temp = true;
		for (int i = 0; i < successTag.length; i++) {
			if (!successTag[i]) {
				temp = false;
				break;
			}
		}
		if (temp) {
			onSuccess();
		}
	}
	
	private void onSuccess() {
		mCacheTask.checkFinish();
		mTaskConfig.mDownloadTaskListener.onSuccess();
	}

	public boolean startDownload() {
		if (!hasStarted) {
			mTaskConfig.mDownloadExecutorService.execute(this);
			return true;
		} else {
			return restartDownload();
		}
	}

	public boolean resumeDownload() {
		// TODO Auto-generated method stub
		return false;
	}

	public boolean pauseDownload() {
		// TODO Auto-generated method stub
		return false;
	}

	public boolean cancelDownload() {
		// TODO Auto-generated method stub
		return false;
	}

	public boolean restartDownload() {
		if (!hasStarted) {
			mTaskConfig.mDownloadExecutorService.execute(this);
			return true;
		} else {
			// TODO
			return false;
		}
	}
}
