/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年3月12日
 */
package org.jabe.neverland.download.core;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * 
 * @Author	LaiLong
 * @Since	2014年3月12日
 */
public abstract class AbstractMessageDeliver implements DownloadRegister {
	
	private MessageListener mEngineMessger;
	
	/**
	 * @param mEngineMessger the mEngineMessger to set
	 */
	public void setEngineMessger(MessageListener mEngineMessger) {
		this.mEngineMessger = mEngineMessger;
	}

	public interface MessageListener {
		public boolean onFire(Message message);
	}
	
	public static class Message {
		public MessageType type;
		public String packageName;
		
		public Message(MessageType type, String packageName) {
			this.type = type;
			this.packageName = packageName;
		}
		
	}

	public static class StatusMessage extends Message {
		
		public int downloadingCount = -1;
		public int changedStatus = -1;
		public Exception exception = null;

		/**
		 * @param type
		 */
		public StatusMessage(String packageName, Exception exception) {
			super(MessageType.STATUS, packageName);
			this.exception = exception;
		}
		
		/**
		 * @param type
		 */
		public StatusMessage(String packageName) {
			super(MessageType.STATUS, packageName);
		}
		
		
	}
	
	public static class ProgressMessage extends Message {
		
		public double added, downloaded, total;

		/**
		 * @param type
		 */
		public ProgressMessage(String packageName) {
			super(MessageType.PROGRESS, packageName);
		}
		
	}
	
	public static enum MessageType {
		PROGRESS, STATUS
	}
	
	protected abstract void push(Message message);
	
	public abstract void start();
	
	public abstract void shutdown();
	
	protected boolean fireMessageToEngine(final Message message) {
		if (mEngineMessger != null) {
			return mEngineMessger.onFire(message);
		}
		return false;
	}
	

	/**
	 * 
	 * @param m
	 */
	protected void fireMessageToClient(final Message m) {
		if (m.type == MessageType.PROGRESS) {
			final ProgressMessage progressMessage = (ProgressMessage) m;
			invokeUpdateProgress(progressMessage.packageName,
					progressMessage.added,
					progressMessage.downloaded,
					progressMessage.total);
		} else if (m.type == MessageType.STATUS) {
			final StatusMessage statusMessage = (StatusMessage) m;
			if (statusMessage.changedStatus >= 0) {
				invokeStatusChanged(statusMessage.packageName, statusMessage.changedStatus);
			} else if (statusMessage.exception != null) {
				invokeFailure(statusMessage.packageName, statusMessage.exception);
			} else if (statusMessage.downloadingCount >= 0) {
				invokeCountChanged(statusMessage.downloadingCount);
			}
		}
	}
	
	private final List<DownloadStatusListener> mDownloadStatusListeners = new ArrayList<DownloadStatusListener>();
	private final List<DownloadProgressListener> mDownloadProgressListeners = new ArrayList<DownloadProgressListener>();
	private final List<DownloadSpeedListener> mDownloadSpeedListeners = new ArrayList<DownloadSpeedListener>();
	
	protected final ReentrantLock mStatusLock = new ReentrantLock();
	protected final ReentrantLock mProgressLock = new ReentrantLock();
	protected final ReentrantLock mSpeedLock = new ReentrantLock();
	

	protected final void invokeUpdateProgress(String packageName, double added,
			double downloaded, double total) {
		mProgressLock.lock();
		for (DownloadProgressListener dl : mDownloadProgressListeners) {
			dl.onUpdateProgress(packageName, added, downloaded, total);
		}
		mProgressLock.unlock();
	}
	
	protected final void invokeCountChanged(int downloadingCount) {
		mStatusLock.lock();
		for (DownloadStatusListener dl : mDownloadStatusListeners) {
			dl.onStatusCountChanged(downloadingCount);
		}
		mStatusLock.unlock();
	}

	protected final void invokeStatusChanged(String packageName, int status) {
		mStatusLock.lock();
		for (DownloadStatusListener dl : mDownloadStatusListeners) {
			dl.onStatusChanged(packageName, status);
		}
		mStatusLock.unlock();
	}

	protected final void invokeFailure(String packageName, Exception e) {
		mStatusLock.lock();
		for (DownloadStatusListener dl : mDownloadStatusListeners) {
			dl.onFailure(packageName, e);
		}
		mStatusLock.unlock();
	}
	
	protected final void invokeSpeedNotify(String message, String packageName, long downloadBytes, long castTime) {
		mSpeedLock.lock();
		for (DownloadSpeedListener dl : mDownloadSpeedListeners) {
			dl.onSpeedNotify(message, packageName, downloadBytes, castTime);
		}
		mSpeedLock.unlock();
	}

	public final void removeListenter(DownloadListener downloadInterface) {
		if (downloadInterface == null) {
			return;
		}
		
		if (downloadInterface instanceof DownloadProgressListener) {
			mProgressLock.lock();
			mDownloadProgressListeners.remove(downloadInterface);
			mProgressLock.unlock();
		} else if (downloadInterface instanceof DownloadStatusListener) {
			mStatusLock.lock();
			mDownloadStatusListeners.remove(downloadInterface);
			mStatusLock.unlock();
		} else if (downloadInterface instanceof DownloadSpeedListener) {
			mSpeedLock.lock();
			mDownloadSpeedListeners.remove(downloadInterface);
			mSpeedLock.unlock();
		}
	}

	public final void registerListener(DownloadListener downloadInterface) {
		if (downloadInterface == null) {
			return;
		}
		
		// may blocking.
		
		if (downloadInterface instanceof DownloadProgressListener) {
			mProgressLock.lock();
			if (!mDownloadProgressListeners.contains(downloadInterface)) {
				mDownloadProgressListeners.add((DownloadProgressListener) downloadInterface);
			}
			mProgressLock.unlock();
		} else if (downloadInterface instanceof DownloadStatusListener) {
			mStatusLock.lock();
			if (!mDownloadStatusListeners.contains(downloadInterface)) {
				mDownloadStatusListeners.add((DownloadStatusListener) downloadInterface);
			}
			mStatusLock.unlock();
		} else if (downloadInterface instanceof DownloadSpeedListener) {
			mSpeedLock.lock();
			if (!mDownloadSpeedListeners.contains(downloadInterface)) {
				mDownloadSpeedListeners.add((DownloadSpeedListener) downloadInterface);
			}
			mSpeedLock.unlock();
		}
	}
	
	public final void pushFinishMessage(String packageName) {
		final StatusMessage statusMessage = new StatusMessage(packageName);
		statusMessage.changedStatus = DownloadStatus.DOWNLOAD_STATUS_FINISHED.ordinal();
		push(statusMessage);
	}
	
	public final void pushExceptionMessage(String packageName, Exception e) {
		final StatusMessage statusMessage = new StatusMessage(packageName);
		statusMessage.exception = e;
		push(statusMessage);
	}
	
	public final void pushStatusChangedMessage(String packageName, int changedStatus) {
		final StatusMessage statusMessage = new StatusMessage(packageName);
		statusMessage.changedStatus = changedStatus;
		push(statusMessage);
	}
	
	public final void pushUpdateProgressMessage(String packageName, double added,
			double downloaded, double total) {
		final ProgressMessage progressMessage = new ProgressMessage(packageName);
		progressMessage.added = added;
		progressMessage.downloaded = downloaded;
		progressMessage.total = total;
		push(progressMessage);
	}
	
	public final void pushDownloadingCountMessage(String packageName, int count) {
		final StatusMessage statusMessage = new StatusMessage(packageName);
		statusMessage.downloadingCount = count;
		push(statusMessage);
	}
}
