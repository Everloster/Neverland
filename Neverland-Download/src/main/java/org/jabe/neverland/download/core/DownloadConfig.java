package org.jabe.neverland.download.core;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.jabe.neverland.download.core.factory.DefaultConfigurationFactory;

public final class DownloadConfig {
	
	public final ExecutorService mTaskExecutor;
	
	public final int mMaxTaskSizeInRunningQueue;
	
	public final DownloadOperationMaper mDownloadOperationMaper;
	public final DownloadEngine mDownloadEngine;
	public final DownloadCacheManager mProgressCacheManager;
	public final AbstractMessageDeliver mMessageDeliver;
	public final String mCacheRootPath;
	
	private static final int DEFAULT_TASK_RUNNING_SIZE = 3;
	
	private DownloadConfig(final Builder builder) {
		mTaskExecutor = builder.mTaskExecutor;
		mMaxTaskSizeInRunningQueue = builder.mMaxTaskSizeInRunningQueue;
		mDownloadOperationMaper = builder.mDownloadOperationMaper;
		mDownloadEngine = builder.mDownloadEngine;
		mProgressCacheManager = builder.mProgressCacheManager;
		mCacheRootPath = builder.mCacheRootPath;
		mMessageDeliver = builder.mMessageDeliver;
	}
	public static class Builder {
		
		private ExecutorService mTaskExecutor;
		private int mMaxTaskSizeInRunningQueue = 0;
		private DownloadOperationMaper mDownloadOperationMaper;
		private DownloadEngine mDownloadEngine;
		private DownloadCacheManager mProgressCacheManager;
		private AbstractMessageDeliver mMessageDeliver;
		private String mCacheRootPath;
		public Builder() {
			
		}
		
		public Builder addTaskExcutor(ExecutorService taskExecutor) {
			this.mTaskExecutor = taskExecutor;
			return this;
		}
		
		public Builder addMaxTaskSizeInRunning(int count) {
			this.mMaxTaskSizeInRunningQueue = count;
			return this;
		}
		
		public Builder addDownloadOperationMapper(DownloadOperationMaper mDownloadOperationMaper) {
			this.mDownloadOperationMaper = mDownloadOperationMaper;
			return this;
		}
		
		public Builder addDownloadCacheManager(DownloadCacheManager cacheManager) {
			this.mProgressCacheManager = cacheManager;
			return this;
		}
		
		public Builder addDownloadEngine(DownloadEngine downloadEngine) {
			this.mDownloadEngine = downloadEngine;
			return this;
		}
		
		public Builder addCacheRootPath(String path) {
			this.mCacheRootPath = path;
			return this;
		}
		
		public Builder addMessageDeliver(AbstractMessageDeliver deliver) {
			this.mMessageDeliver = deliver;
			return this;
		}
		
		public DownloadConfig build() {
			initEmptyConfig();
			return new DownloadConfig(this);
		}
		
		private void initEmptyConfig() {
			if (mMaxTaskSizeInRunningQueue == 0) {
				mMaxTaskSizeInRunningQueue = DEFAULT_TASK_RUNNING_SIZE;
			}
			if (mTaskExecutor == null) {
				mTaskExecutor = Executors.newCachedThreadPool();
			}
			if (mCacheRootPath == null || mCacheRootPath.equals("")) {
				// TODO  to get default cache root path by system properties.
				mCacheRootPath = "C://Users//Administrator//Desktop//";
			}
			if (mProgressCacheManager == null) {
				mProgressCacheManager = DefaultConfigurationFactory.getDefaultCacheManager(mCacheRootPath);
			}
			
			if (mMessageDeliver == null) {
				mMessageDeliver = new DefaultMessageDeliver();
			}
			
			if (mDownloadEngine == null) {
				mDownloadEngine = DefaultConfigurationFactory.getDefaultDownloadEngine(mProgressCacheManager, mTaskExecutor, mMessageDeliver, mMaxTaskSizeInRunningQueue);
			}
			if (mDownloadOperationMaper == null) {
				mDownloadOperationMaper = DefaultConfigurationFactory.getDefaultMaper(mDownloadEngine);
			}
		}
	}
}
