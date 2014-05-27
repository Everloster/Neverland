package org.jabe.neverland.download.core.engine;

import java.util.concurrent.ExecutorService;

import org.jabe.neverland.download.core.AbstractMessageDeliver;
import org.jabe.neverland.download.core.IODownloader;

public class TaskConfig {
	
	public final DownloadCacheInvoker mCacheInvoker;
	public final ExecutorService mDownloadExecutorService;
	public final IODownloader mDownloader;
	public final AbstractMessageDeliver mMessageDeliver;
	
	private TaskConfig(final Builder builder) {
		mCacheInvoker = builder.mCacheTask;
		mDownloadExecutorService = builder.mDownloadExecutorService;
		mDownloader = builder.mDownloader;
		mMessageDeliver = builder.mMessageDeliver;
	}
	
	public static class Builder {
		
		private DownloadCacheInvoker mCacheTask;
		private ExecutorService mDownloadExecutorService;
		private IODownloader mDownloader;
		private AbstractMessageDeliver mMessageDeliver;
		
		public Builder() {
		}
		
		public Builder addCacheInvoker(final DownloadCacheInvoker cacheTask) {
			this.mCacheTask = cacheTask;
			return this;
		}
		
		public Builder addMessageDeliver(final AbstractMessageDeliver messageDeliver) {
			this.mMessageDeliver = messageDeliver;
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
