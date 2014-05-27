package org.jabe.neverland.download.core.engine;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.locks.ReentrantLock;

import org.jabe.neverland.download.core.AbstractMessageDeliver;
import org.jabe.neverland.download.core.DownloadCacheManager;
import org.jabe.neverland.download.core.DownloadEngine;
import org.jabe.neverland.download.core.DownloadInfo;
import org.jabe.neverland.download.core.DownloadListenerWraper;
import org.jabe.neverland.download.core.IODownloader;
import org.jabe.neverland.download.core.AbstractMessageDeliver.Message;
import org.jabe.neverland.download.core.AbstractMessageDeliver.MessageListener;
import org.jabe.neverland.download.core.AbstractMessageDeliver.MessageType;
import org.jabe.neverland.download.core.AbstractMessageDeliver.StatusMessage;
import org.jabe.neverland.download.core.DownloadStatus;
import org.jabe.neverland.download.log.Logger;

public class CacheDownloadEngine extends DownloadListenerWraper implements
		DownloadEngine {

	protected final Map<String, AbstractMessageTask> mDownloadTaskMap = new HashMap<String, AbstractMessageTask>();
	protected final ReentrantLock mTaskLock = new ReentrantLock();
	protected final DownloadCacheManager mProgressCacheManager;
	protected final ExecutorService mDownloadExecutor;
	protected volatile IODownloader mIODownloader = null;
	
	private final int mMaxDownloadCount;
	
	private final ConcurrentLinkedQueue<DownloadInfo> mWaitingTask = new ConcurrentLinkedQueue<DownloadInfo>();

	private AbstractMessageDeliver mMessageDeliver;

	public CacheDownloadEngine(DownloadCacheManager mProgressCacheManager,
			ExecutorService executorService,
			AbstractMessageDeliver messageDeliver, int coreCount) {
		super(null);
		this.mProgressCacheManager = mProgressCacheManager;
		this.mDownloadExecutor = executorService;
		this.mMessageDeliver = messageDeliver;
		mMessageDeliver.setEngineMessger(mGlobalMessager);
		mMaxDownloadCount = coreCount;
	}

	private MessageListener mGlobalMessager = new MessageListener() {

		@Override
		public boolean onFire(Message m) {
			if (m.type == MessageType.STATUS) {
				final StatusMessage statusMessage = (StatusMessage) m;
				if (statusMessage.changedStatus >= 0
						&& (statusMessage.changedStatus == DownloadStatus.DOWNLOAD_STATUS_FINISHED
								.ordinal()
								|| statusMessage.changedStatus == DownloadStatus.DOWNLOAD_STATUS_PAUSED
										.ordinal()
								|| statusMessage.changedStatus == DownloadStatus.DOWNLOAD_STATUS_CANCEL
										.ordinal() || statusMessage.changedStatus == DownloadStatus.DOWNLOAD_STATUS_FAILED
								.ordinal())) {

					removeTaskByPackageName(statusMessage.packageName);
					mMessageDeliver.pushDownloadingCountMessage(m.packageName,
							getCurrentDownloadingCount());
					
					triggerRunWaitTask();

					return false;
				}

				if (statusMessage.downloadingCount >= 0) {
					return false;
				}
				
				if (statusMessage.changedStatus == DownloadStatus.DOWNLOAD_STATUS_WAIT
								.ordinal()) {
					return false;
				}
				

				if (statusMessage.changedStatus >= 0
						&& statusMessage.changedStatus == DownloadStatus.DOWNLOAD_STATUS_PREPARE
								.ordinal()) {
					mMessageDeliver.pushDownloadingCountMessage(m.packageName,
							getCurrentDownloadingCount());
				}
			}
			
			//  过滤不在下载状态中的消息
			
			mTaskLock.lock();
			boolean taskExist = false;
			if (mDownloadTaskMap.containsKey(m.packageName)) {
				final DownloadTask downloadTask = mDownloadTaskMap
						.get(m.packageName);
				if (downloadTask.isDownloading()) {
					taskExist = true;
				}
			}
			mTaskLock.unlock();
			return !taskExist;
		}
	};

	private int getCurrentDownloadingCount() {
		int count = 0;
		mTaskLock.lock();
		count = mDownloadTaskMap.size();
		mTaskLock.unlock();
		return count;
	}

	@Override
	public boolean startDownload(DownloadInfo downloadInfo) {
		return startD(downloadInfo);
	}

	@Override
	public boolean resumeDownload(DownloadInfo downloadInfo) {
		return startD(downloadInfo);
	}

	@Override
	public boolean pauseDownload(DownloadInfo downloadInfo) {
		return pauseD(downloadInfo);
	}

	@Override
	public boolean cancelDownload(DownloadInfo downloadInfo) {
		return cancelD(downloadInfo);
	}

	@Override
	public boolean restartDownload(DownloadInfo downloadInfo) {
		return startD(downloadInfo);
	}

	protected boolean cancelD(DownloadInfo downloadInfo) {
		
		if (mWaitingTask.contains(downloadInfo)) {
			mWaitingTask.remove(downloadInfo);
			mMessageDeliver.pushStatusChangedMessage(downloadInfo.getmPackageName(), DownloadStatus.DOWNLOAD_STATUS_CANCEL.ordinal());
			return true;
		}
		
		mTaskLock.lock();
		if (mDownloadTaskMap.containsKey(downloadInfo.getmPackageName())) {
			final DownloadTask downloadTask = mDownloadTaskMap.get(downloadInfo
					.getmPackageName());
			downloadTask.cancel();
			mTaskLock.unlock();
			return true;
		} else {
			mTaskLock.unlock();
			return false;
		}
	}

	protected boolean pauseD(DownloadInfo downloadInfo) {
		
		if (mWaitingTask.contains(downloadInfo)) {
			mWaitingTask.remove(downloadInfo);
			mMessageDeliver.pushStatusChangedMessage(downloadInfo.getmPackageName(), DownloadStatus.DOWNLOAD_STATUS_PAUSED.ordinal());
			return true;
		}
		
		mTaskLock.lock();
		if (mDownloadTaskMap.containsKey(downloadInfo.getmPackageName())) {
			final DownloadTask downloadTask = mDownloadTaskMap.get(downloadInfo
					.getmPackageName());
			downloadTask.pause();
			mTaskLock.unlock();
			return true;
		} else {
			mTaskLock.unlock();
			return false;
		}
	}

	protected boolean startD(DownloadInfo downloadInfo) {
		if (downloadInfo == null) {
			return false;
		}
		// check whether is already downloaded.
		if (mProgressCacheManager.isDownloadFinished(downloadInfo)) {
			Logger.i("The pacakge is already exist.");
			mMessageDeliver.pushFinishMessage(downloadInfo.getmPackageName());
			return true;
		}

		// if the task was already in the map.
		mTaskLock.lock();
		if (mDownloadTaskMap.containsKey(downloadInfo.getmPackageName())) {
			final DownloadTask downloadTask = mDownloadTaskMap.get(downloadInfo
					.getmPackageName());
			mTaskLock.unlock();
			return downloadTask.start();
		} else {
			return trafficControlStart(downloadInfo);
		}

	}

	/**
	 * 
	 * @param downloadInfo
	 * @return
	 */
	private boolean trafficControlStart(DownloadInfo downloadInfo) {
		if (mDownloadTaskMap.size() >= mMaxDownloadCount) {
			mTaskLock.unlock();
			if (mWaitingTask.contains(downloadInfo)) {
				return false;
			}
			mWaitingTask.add(downloadInfo);
			mMessageDeliver.pushStatusChangedMessage(downloadInfo.getmPackageName(), DownloadStatus.DOWNLOAD_STATUS_WAIT.ordinal());
			return true;
		} else {
			final DownloadCacheInvoker cacheInvoker = new DownloadCacheInvoker(
					mProgressCacheManager, downloadInfo);
			final TaskConfig taskConfig = new TaskConfig.Builder()
					.addCacheInvoker(cacheInvoker)
					.addDownloadExecutor(mDownloadExecutor)
					.addMessageDeliver(mMessageDeliver)
					.addDownloader(
							mIODownloader != null ? mIODownloader
									: getDefaultDownloader()).build();
			final AbstractMessageTask downloadTask = new MultiThreadTask(
					taskConfig);
			mDownloadTaskMap.put(downloadInfo.getmPackageName(), downloadTask);
			mTaskLock.unlock();
			Logger.i("Begin to download the package : "
					+ downloadInfo.getmPackageName());
			return downloadTask.start();
		}
		
	}
	
	private void triggerRunWaitTask() {
		mTaskLock.lock();
		if (mDownloadTaskMap.size() < mMaxDownloadCount) {
			mTaskLock.unlock();
			final DownloadInfo downloadInfo = mWaitingTask.poll();
			if (downloadInfo != null) {
				startD(downloadInfo);
			}
		} else {
			mTaskLock.unlock();
		}
	}

	private void removeTaskByPackageName(String name) {
		mTaskLock.lock();
		if (mDownloadTaskMap.containsKey(name)) {
			final AbstractMessageTask multiThreadTask = mDownloadTaskMap
					.get(name);
			multiThreadTask.clear();
			mDownloadTaskMap.remove(name);
		}
		mTaskLock.unlock();
	}

	private IODownloader defaultDownloader = null;

	private IODownloader getDefaultDownloader() {
		if (defaultDownloader == null) {
			defaultDownloader = new BaseIODownloader();
		}
		return defaultDownloader;
	}

	@Override
	public void setIoDownloader(IODownloader downloader) {
		this.mIODownloader = downloader;
	}
}
