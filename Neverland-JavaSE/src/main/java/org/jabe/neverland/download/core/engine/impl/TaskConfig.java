package org.jabe.neverland.download.core.engine.impl;

import java.util.concurrent.ExecutorService;

import org.jabe.neverland.download.core.cache.DownloadCacheTask;

public class TaskConfig {
	
	public final DownloadCacheTask mCacheTask;
	public final DownloadTaskListener mDownloadTaskListener;
	public final ExecutorService mDownloadExecutorService;
	public final IODownloader mDownloader;
	
	private TaskConfig(final Builder builder) {
		mCacheTask = builder.mCacheTask;
		mDownloadTaskListener = builder.mDownloadTaskListener;
		mDownloadExecutorService = builder.mDownloadExecutorService;
		mDownloader = builder.mDownloader;
	}
	
	public static class Builder {
		
		private DownloadCacheTask mCacheTask;
		private DownloadTaskListener mDownloadTaskListener;
		private ExecutorService mDownloadExecutorService;
		private IODownloader mDownloader;
		
		public Builder() {
		}
		
		public Builder addCacheTask(final DownloadCacheTask cacheTask) {
			this.mCacheTask = cacheTask;
			return this;
		}
		
		public Builder addDownloadTaskListener(final DownloadTaskListener downloadTaskListener) {
			this.mDownloadTaskListener = downloadTaskListener;
			return this;
		}
		
		public Builder addDownloadExecutor(final ExecutorService executorService) {
			this.mDownloadExecutorService = executorService;
			return this;
		}
		
		public Builder addDownloader(final IODownloader downloader) {
			this.mDownloader = downloader;
			return this;
		}
		
		public TaskConfig build() {
			return new TaskConfig(this);
		}
	}
}
