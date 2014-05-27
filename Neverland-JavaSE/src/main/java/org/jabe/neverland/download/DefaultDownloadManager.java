package org.jabe.neverland.download;

import java.io.File;
import java.io.RandomAccessFile;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DefaultDownloadManager extends DownloadManager {

	public static void main(String[] args) {
		final String url = "http://storefs.nearme.com.cn/uploadFiles/Pfiles/201209/a31ee18e4cb741dfa3f8bced1ce6c29a.apk";
		DefaultDownloadManager defaultDownloadManager = new DefaultDownloadManager("H:/");
		defaultDownloadManager.startDownload(url, "com.boyaa.lordland.nearme.gamecenter");
	}

	private File mDownloadPathFile;

	private String mDownloadPathString;

	private ExecutorService mExecutorService = Executors.newCachedThreadPool();

	public DefaultDownloadManager(String path) {
		mDownloadPathString = path;
		mDownloadPathFile = new File(path);
	}

	private Map<String, TaskAssign> mTaskMap = new HashMap<String, TaskAssign>();

	@Override
	public void startDownload(String url, String tag) {
		if (hasDownloadingFile(url, tag)) {
			resumeDownload(tag);
			return;
		}
		if (!isDownloading(tag)) {
			startNewTask(url, tag);
		}
	}

	private void startNewTask(String url, String tag) {
		final Task task = new Task();
		task.setDownURL(url);
		task.setPackageName(tag);
		task.setSaveFile(mDownloadPathString + generateFileName(url, tag));
		task.setSectionCount(1);
		task.setWorkerCount(1);
		task.setBufferSize(8 * 1024);
		final TaskAssign ta = new TaskAssign(mExecutorService);
		final TaskListener tl = new TaskListener(tag) {
			
			@Override
			public void onUpdateProgress(double added, double downloaded, double total) {
				Loger.log(TAG, "Progress: " + (downloaded/total) * 100);
			}
			
			@Override
			public void onSuccess() {
				sendStatusToListener(DownloadStatus.DOWNLOAD_STATUS_FINISHED, getTag());
			}
			
			@Override
			public void onPreTask() {
				sendStatusToListener(DownloadStatus.DOWNLOAD_STATUS_STARTED, getTag());
			}
			
			@Override
			public void onFileExist(File file) {
				Loger.log(TAG, "file already exist !");
				sendStatusToListener(DownloadStatus.DOWNLOAD_STATUS_FINISHED, getTag());
			}
			
			@Override
			public void onFailure(Exception e) {
				Loger.log(TAG, e.getMessage());
			}
		};;;
		ta.setTaskListener(tl);
		ta.work(task);
		mTaskMap.put(tag, ta);
	}
	
	private void sendStatusToListener(DownloadStatus status, String tag) {
		if (status == DownloadStatus.DOWNLOAD_STATUS_FINISHED) {
			Loger.log(TAG, "下载完成啦!");
		}
	}

	public Task getCacheDownloadTask(String url, String tag) {
		File file = new File(mDownloadPathFile, generateFileName(url, tag));
		if (file.isFile() && file.exists()) {
			Task task = new Task();
			RandomAccessFile raf = null;
			try {
				raf = new RandomAccessFile(file, "rw");
				task.read(raf);
			} catch (Exception e) {
				file.delete();
				task = null;
			} finally {
				try {
					if (raf != null) {
						raf.close();
					}
				} catch (Exception e) {
				}
			}
			return task;
		} else {
			return null;
		}
	}

	@Override
	public boolean resumeDownload(String tag) {
		if (isDownloading(tag)) {
			mTaskMap.get(tag).resumeWork();
			return true;
		} else {
			return false;
		}
	}

	@Override
	public boolean pauseDownload(String tag) {
		if (isDownloading(tag)) {
			mTaskMap.get(tag).pauseWork();
			return true;
		} else {
			return false;
		}
	}

	@Override
	public boolean cancelDownload(String tag) {
		if (isDownloading(tag)) {
			mTaskMap.get(tag).cancel();
			return true;
		} else {
			return false;
		}
	}

	@Override
	public boolean restartDownload(String tag) {
		return false;
	}

	@Override
	public boolean hasDownloadingFile(String url, String tag) {
		return getCacheDownloadTask(url, tag) == null ? false : true;
	}

	@Override
	public boolean isDownloading(String tag) {
		return mTaskMap.containsKey(tag);
	}

	@Override
	public boolean hasFinished(String url, String tag) {
		try {
			String path = mDownloadPathString + generateFileName(url, tag);
			File file = new File(path);
			if (file.exists()) {
				return true;
			} else {
				return false;
			}
		} catch (Exception e) {
			return false;
		}
	}

	@Override
	public String generateFileName(String url, String tag) {
		if (url == null)
			return null;
		int p = url.lastIndexOf("/");
		return url.substring(p + 1);
	}

}
