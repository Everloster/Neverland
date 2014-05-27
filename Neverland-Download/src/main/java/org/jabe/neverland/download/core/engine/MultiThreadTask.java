package org.jabe.neverland.download.core.engine;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import org.jabe.neverland.download.log.Logger;
import org.jabe.neverland.download.util.IoUtils;


public class MultiThreadTask extends AbstractMessageTask {
	
	private static final int BUFFER_SIZE = 16 * 1024;
	
	
	public MultiThreadTask(TaskConfig mTaskConfig) {
		super(mTaskConfig);
	}

	private volatile Boolean[] successTag;
	
	@Override
	public void run() {
		keepRunning = true;
		// task life cycle
		onPreTask();
		
		try {
			// whatever, need to get content length to check the task in cache
			if (mCacheDownloadInfo.mContentLength == 0) {
				mCacheDownloadInfo.mContentLength = getContentLength(mCacheDownloadInfo.mDownloadInfo.getmDownloadUrl());
			}
			if (mCacheInvoker.isInCache()) {
				mCacheInvoker.readFromCache();
			} else {
				mCacheInvoker.saveToCache();
			}
		} catch (IOException e) {
			triggerIOException(e);
			return;
		}
		
		final int secCount = mCacheDownloadInfo.mSectionCount;

		// task life cycle
		onBeforeExecute();
		
		if (mCacheDownloadInfo.mDownloadedLength == mCacheDownloadInfo.mContentLength && mCacheDownloadInfo.mContentLength > 0) {
			success();
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
		} catch (IOException e) {
			triggerIOException(e);
		}
	}
	
	private void doSingleWork() throws IOException {
		doRealDownload(1, mCacheDownloadInfo.mContentLength, mCacheDownloadInfo.mDownloadedLength, mCacheDownloadInfo.mContentLength);
	}
	
	private void doMultipleWork() throws FileNotFoundException {
		final int secCount = mCacheDownloadInfo.mSectionCount;
//		mCacheAccessFile = new RandomAccessFile(mCacheTask.generateCacheSaveFullPath(), "rw");
		for (int j = 0; j < secCount; j++) {
			final long length = mCacheDownloadInfo.mContentLength;
			final long start = mCacheDownloadInfo.mSectionsOffset[j];
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
			if (realStart == endF) {
				triggerSuccess(s);
				continue;
			}
			getExecutorService().execute(new Runnable() {
				
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
		final InputStream is = getStream(mCacheDownloadInfo.mDownloadInfo.getmDownloadUrl(), null, sb);
		try {
			byte[] bytes = new byte[BUFFER_SIZE];
			while (keepRunning) {
				int count = is.read(bytes, 0, BUFFER_SIZE);
				if (count == -1) {
					break;
				}
				if (keepRunning) {
					mCacheInvoker.updateSectionProgress(bytes, sectionNo, count);
					onUpdateProgress(count, mCacheDownloadInfo.mDownloadedLength, mCacheDownloadInfo.mContentLength);
				}
			}
			triggerSuccess(sectionNo);
		} catch(IOException e) {
			e.printStackTrace();
			triggerSectionFailure(sectionNo);
		} finally {
			IoUtils.closeSilently(is);
		}
	}
	
	/**
	 * get content length by url connection.
	 * 
	 * @param url
	 * @return
	 * @throws IOException
	 */
	private long getContentLength(String url) throws IOException {
		URL u = new URL(url);
		HttpURLConnection conn = (HttpURLConnection) u.openConnection();
		long l = conn.getContentLength();
		conn.disconnect();
		return l;
	}
	
	private void triggerSectionFailure(int section) {
		Logger.i(getPackageName() + " " + section + " section download failure.");
		triggerIOException(null);
	}
	
	private void triggerIOException(IOException e) {
		stop();
		if (e != null) {
			onFailure(e);
		} else {
			onFailure(new IOException("File or Network Exception."));
		}
	}
	
	private void triggerSuccess(int sectionNo) {
		if (mCacheDownloadInfo.isSectionOk(sectionNo)) {
			successTag[sectionNo - 1] = true;
			Logger.i(getPackageName() + " section num : " + sectionNo + " ok.");
			checkAllSuccess();	
		} else {
			triggerSectionFailure(sectionNo);
		}
	}

	private void checkAllSuccess() {
		boolean success = true;
		for (int i = 0; i < successTag.length; i++) {
			if (!successTag[i]) {
				success = false;
				break;
			}
		}
		if (success) {
			success();
		}
	}
	
	private void success() {
		mCacheInvoker.completeCacheTask();
		super.onSuccess();
	}

	@Override
	public boolean start() {
		if (!keepRunning) {
			getExecutorService().execute(this);
		}
		return true;
	}

	@Override
	public boolean resume() {
		return start();
	}

	@Override
	public boolean cancel() {
		stop();
		clearCache();
		super.onCancelTask();
		return true;
	}

	@Override
	public boolean pause() {
		stop();
		super.onPauseTask();
		return true;
	}

	@Override
	public void clearCache() {
		mCacheInvoker.clearCache();
	}

	@Override
	public void stop() {
		keepRunning = false;
	}
	
	@Override
	public boolean isDownloading() {
		return keepRunning;
	}
}
