/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年3月6日
 */
package org.jabe.neverland.download.core.engine;

import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;

import org.jabe.neverland.download.core.AbstractMessageDeliver;
import org.jabe.neverland.download.core.DownloadStatus;
import org.jabe.neverland.download.core.IODownloader;

/**
 * message deliver task
 * 
 * @Author LaiLong
 * @Since 2014年3月6日
 */
public abstract class AbstractMessageTask extends CacheDownloadTask implements
		IODownloader {

	private IODownloader mDownloader;

	private ExecutorService mExecutorService;
	
	private AbstractMessageDeliver mMessageDeliver;
	
	public void clear() {
		super.clear();
		mDownloader = null;
		mExecutorService = null;
		mMessageDeliver = null;
	}

	/**
	 * @param cacheInvoker
	 * @param listener
	 */
	public AbstractMessageTask(TaskConfig taskConfig) {
		super(taskConfig.mCacheInvoker);
		this.mDownloader = taskConfig.mDownloader;
		this.mMessageDeliver = taskConfig.mMessageDeliver;
		this.mExecutorService = taskConfig.mDownloadExecutorService;
	}

	protected ExecutorService getExecutorService() {
		return mExecutorService;
	}

	protected void onSuccess() {
		mMessageDeliver.pushFinishMessage(getPackageName());
	}

	protected void onFailure(Exception e) {
		mMessageDeliver.pushExceptionMessage(getPackageName(), e);
	}

	protected void onPreTask() {
		mMessageDeliver.pushStatusChangedMessage(getPackageName(), DownloadStatus.DOWNLOAD_STATUS_PREPARE.ordinal());
	}

	protected void onResumeTask() {
		mMessageDeliver.pushStatusChangedMessage(getPackageName(), DownloadStatus.DOWNLOAD_STATUS_RESUME.ordinal());
	}

	protected void onPauseTask() {
		mMessageDeliver.pushStatusChangedMessage(getPackageName(), DownloadStatus.DOWNLOAD_STATUS_PAUSED.ordinal());
	}

	protected void onBeforeExecute() {
		mMessageDeliver.pushStatusChangedMessage(getPackageName(), DownloadStatus.DOWNLOAD_STATUS_STARTED.ordinal());
	}

	protected void onCancelTask() {
		mMessageDeliver.pushStatusChangedMessage(getPackageName(), DownloadStatus.DOWNLOAD_STATUS_CANCEL.ordinal());
	}

	protected void onUpdateProgress(double added, double downloaded, double total) {
		mMessageDeliver.pushUpdateProgressMessage(getPackageName(), added, downloaded, total);
	}

	@Override
	public InputStream getStream(String imageUri, Object extra, SizeBean sb)
			throws IOException {
		return mDownloader.getStream(imageUri, extra, sb);
	}

}
