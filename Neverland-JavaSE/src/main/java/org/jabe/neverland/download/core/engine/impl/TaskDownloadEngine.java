package org.jabe.neverland.download.core.engine.impl;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;

import org.jabe.neverland.download.core.DownloadEngine;
import org.jabe.neverland.download.core.DownloadInfo;
import org.jabe.neverland.download.core.cache.DownloadCacheManager;
import org.jabe.neverland.download.core.cache.DownloadCacheTask;

public class TaskDownloadEngine implements DownloadEngine {

	protected final Map<String, DownloadTask> mDownloadTaskMap = Collections.synchronizedMap(new HashMap<String, DownloadTask>());
	protected final DownloadCacheManager mProgressCacheManager;
	protected final ExecutorService mDownloadExecutor;

	public TaskDownloadEngine(DownloadCacheManager mProgressCacheManager,
			ExecutorService executorService) {
		this.mProgressCacheManager = mProgressCacheManager;
		this.mDownloadExecutor = executorService;
	}

	@Override
	public boolean startDownload(DownloadInfo downloadInfo) {
		// TODO Auto-generated method stub
		return startD(downloadInfo);
	}

	@Override
	public boolean resumeDownload(DownloadInfo downloadInfo) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean pauseDownload(DownloadInfo downloadInfo) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean cancelDownload(DownloadInfo downloadInfo) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean restartDownload(DownloadInfo downloadInfo) {
		// TODO Auto-generated method stub
		return false;
	}

	protected boolean startD(DownloadInfo downloadInfo) {
		if (mDownloadTaskMap.containsKey(downloadInfo.getmPackageName())) {
			final DownloadTask downloadTask = mDownloadTaskMap.get(downloadInfo
					.getmPackageName());
			return downloadTask.restartDownload();
		} else {
			final DownloadCacheTask cacheTask = new DownloadCacheTask(mProgressCacheManager, downloadInfo);
			final TaskConfig taskConfig = new TaskConfig.Builder()
					.addCacheTask(cacheTask)
					.addDownloadExecutor(mDownloadExecutor)
					.addDownloadTaskListener(mDownloadTaskListener)
					.addDownloader(new BaseIODownloader())
					.build();
			final DownloadTask downloadTask = new DownloadTask(taskConfig);
			mDownloadTaskMap.put(downloadInfo.getmPackageName(), downloadTask);
			return downloadTask.startDownload();
		}
	}
	
	protected DownloadTaskListener mDownloadTaskListener = new DownloadTaskListener("") {
		
		@Override
		public void onUpdateProgress(double added, double downloaded, double total) {
			// TODO Auto-generated method stub
			System.out.println("Downloaded Percent : " + downloaded/total);
//			System.out.println("changed by current thread " + Thread.currentThread().getName() + " " + downloaded + "/" + total);
			
		}
		
		@Override
		public void onSuccess() {
			// TODO Auto-generated method stub
			System.out.println("Download Success ");
		}
		
		@Override
		public void onFailure(Exception e) {
			// TODO Auto-generated method stub
			
		}
	};
}
