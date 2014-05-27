package org.jabe.neverland.download;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class TaskAssign {
	public static final String TAG = "TaskAssign";

	public static final String APPEND_TASKFILE = ".r_task";
	public static final String APPEND_SAVEFIlE = ".r_save";
	private TaskListener taskListener;
	private volatile boolean keepWorking = true;
	private volatile boolean needWait = false;
	private Task mTask;
	private Boolean[] successTag;
	private RandomAccessFile taskRandomFile;
	private RandomAccessFile downRandomFile;
	private File taskFile;
	private File saveFile;
	private boolean useDelay = true;
	private int currnetPercent = 0;
	private static final int PERCENT_UNIT = 1;

	public void work(final Task task) {
		mTask = task;
		Thread th = new Thread(new Runnable() {

			@Override
			public void run() {
				mTask = task;
				if (taskListener == null || task.getContentLength() < 0) {
					Loger.log(TAG,
					"task listener can not be null or task content length can not < 0");
					return;
				}

				// TaskListener lifecycle
				taskListener.onPreTask();

				File file = new File(task.getSaveFile());
				if (file.exists()) {
					checkRandomFileClose();
					taskListener.onFileExist(file);
					return;
				}

				// 任务描述文件

				taskFile = new File(task.getSaveFile() + APPEND_TASKFILE);

				// 真正下载的文件

				saveFile = new File(task.getSaveFile() + APPEND_SAVEFIlE);

				boolean taskFileExist = taskFile.exists();

				checkRandomFileClose();

				try {

					taskRandomFile = new RandomAccessFile(taskFile, "rw");

					downRandomFile = new RandomAccessFile(saveFile, "rw");

					long rtnLen = task.getContentLength();

					Loger.log(TAG, "get length : " + (double) rtnLen / 1024
							+ "K");

					if (!taskFileExist) {

						// 如果文件不存在，就初始化任务文件和下载文件

						task.setContentLength(rtnLen);

						initTaskFile(taskRandomFile, task);

						//如果文件很大,会用时很久,也可能存储空间不够.
						//存储空间不够抛出的是IOException, 文件会被创建, 大小为剩余容量, 所以创建失败后应该删除.
						Loger.log(TAG, "begin to create save file , size:" + rtnLen);
						try {
							downRandomFile.setLength(rtnLen);
						} catch (Exception e) {
							throw new StorageFullException(e.getMessage());
						}
						Loger.log(TAG, "end to create save file , size:" + rtnLen);

					} else {

						// 检查 任务文件 的合法性
						if(!validateTaskFile(taskRandomFile)) {
							checkRandomFileClose();
							taskFile.delete();
							saveFile.delete();
							throw new ReadTaskFileException("while validateTaskFile");
						}

//						taskListener.resumeTask();

						task.read(taskRandomFile);
					}

					int secCount = task.getSectionCount();

					// 分配线程去下载，这里用到线程池

					taskListener.onBeforeExecute();

					if (secCount > 1) {
						doMultipleWork(task, secCount);
					} else {
						doSingleWork();
					}

				} catch (Exception e) {
					e.printStackTrace();
					Loger.log(TAG, "A exception occured at main work thread: " + e.toString());
					handMainException(e);
				}
			}
		});
		th.start();
	}
	
	private void handMainException(Exception e) {
		stopWork();
		checkRandomFileClose();
		deleteSaveAndTaskFile();
		taskListener.onFailure(e);
	}

	private void deleteSaveAndTaskFile() {
		if (saveFile != null) {
			saveFile.delete();
		}
		if (taskFile != null) {
			taskFile.delete();
		}
	}
	private boolean validateTaskFile(RandomAccessFile taskRandomFile) {
		try {
			if (taskRandomFile != null && taskRandomFile.length() == 0) {
				return false;
			}
		} catch (IOException e) {
			return false;
		}
		return true;
	}

	private void doSingleWork() {
		final RandomAccessFile f1 = taskRandomFile;
		final RandomAccessFile f2 = downRandomFile;
		final long length = mTask.getContentLength();
		final long start = mTask.getDownloadedLength();
		final long end = mTask.getContentLength();
		if (start == end) {
			success();
			return;
		}
		Thread worker = new Thread(new Runnable() {
			@Override
			public void run() {
				downSingle(f1, f2, mTask, length, start, end);
			}
		});
		worker.start();
	}
	
	public Task getTask() {
		return mTask;
	}

	private void doMultipleWork(Task task, int secCount) {
		ExecutorService es = Executors
				.newFixedThreadPool(task.getWorkerCount());

		Loger.log(TAG, "start worker : " + secCount);
		successTag = new Boolean[secCount];
		for (int i = 0; i < successTag.length; i++) {
			successTag[i] = false;
		}
		for (int i = 0; i < secCount; i++) {
			final int j = i;

			final Task t = task;

			final RandomAccessFile f1 = taskRandomFile;

			final RandomAccessFile f2 = downRandomFile;

			final long length = t.getContentLength();

			final long start = t.getSectionsOffset()[j];

			long endt = -1;

			long per = 0;

			// 这里要注意一下，这里是计算当前块的长度

			if (j < t.getSectionCount() - 1) {

				per = t.getContentLength() / t.getSectionCount();

				endt = per * (j + 1);

			} else {

				endt = t.getContentLength();

				per = endt - start;

			}

			final long end = endt;

			if (start >= end) {
				triggerSuccess(j);
				Loger.log(TAG, "Section has finished before. " + j);
				continue;
			}

			es.execute(new Runnable() {
				public void run() {
					down(f1, f2, t, start, end, length, j);
				}

			});
		}
	}

	private void success() {
		checkRandomFileClose();
		if (!taskFile.delete()) {
			taskFile = new File(mTask.getSaveFile() + APPEND_TASKFILE);
			if (taskFile.exists() && taskFile.isFile()) {
				if (!taskFile.delete()) {
					Loger.log(TAG, "delete file failure!");
				} else {

				}
			}
		}
		if (!saveFile.renameTo(new File(mTask.getSaveFile()))) {
			saveFile = new File(mTask.getSaveFile() + APPEND_SAVEFIlE);
			if (saveFile.exists()) {
				if (!saveFile.renameTo(new File(mTask.getSaveFile()))) {
					Loger.log(TAG, "rename file failure!");
				} else {

				}
			}
		}
		taskListener.onSuccess();
	}

	private void checkRandomFileClose() {
		if (taskRandomFile != null) {
			try {
				taskRandomFile.close();
			} catch (IOException e) {
			}
			taskRandomFile = null;
		}
		if (downRandomFile != null) {
			try {
				downRandomFile.close();
			} catch (IOException e) {

			}
			downRandomFile = null;
		}
	}

	/**
	 * open connection to download data from start to end.
	 * 
	 * @param taskRandomFile
	 * @param downRandomFile
	 * @param task
	 * @param start
	 * @param end
	 * @param length
	 * @param sectionNo
	 *            index
	 * @throws IOException
	 */
	private void down(RandomAccessFile taskRandomFile,
			RandomAccessFile downRandomFile, Task task, long start, long end,
			long length, int sectionNo) {
		HttpURLConnection conn = null;
		try {
			long oldTime = System.currentTimeMillis();

			// 这里我用HttpURLConnection下载，你也可以用HttpClient或者自己实现一个Http协议（不过貌似没有必要）

			URL u = new URL(task.getDownURL());

			conn = (HttpURLConnection) u.openConnection();
			initHeaders(conn);

			String range = "bytes=" + start + "-" + (end - 1);

			conn.setRequestProperty("Range", range);

			conn.connect();

			if (conn.getResponseCode() != 206) {

				throw new Exception("conn.getResponseCode() != 206");

			}

			if (conn.getContentLength() != (end - start)) {

				throw new Exception("conn.getContentLength() != (end - start)");

			}

			InputStream is = conn.getInputStream();

			byte[] temp = new byte[task.getBufferSize()];

			BufferedInputStream bis = new BufferedInputStream(is, temp.length);

			int readed = 0;

			while ((readed = bis.read(temp)) > 0 && keepWorking) {
				long offset = task.getSectionsOffset()[sectionNo];
				synchronized (task) {
					// 下载之后顺便更新描述文件，你可能会发现这里效率比较低，在一个线程同步里进行两次文件操作。你可以自己实现一个缓冲写。

					downRandomFile.seek(offset);

					downRandomFile.write(temp, 0, readed);

					offset += readed;

					task.getSectionsOffset()[sectionNo] = offset;

					task.updateDownloadedLength(readed);

					task.writeOffset(taskRandomFile);

					taskListener.onUpdateProgress(readed,
							task.getDownloadedLength(), length);
					
					if (needWait) {
						task.wait();
					}
				}
			}
			if (keepWorking) {
				triggerSuccess(sectionNo);
			}
			if (conn != null) {
				conn.disconnect();
				conn = null;
			}
			Loger.log(TAG, "Section finished. " + sectionNo + "cast: "
					+ (System.currentTimeMillis() - oldTime));
		} catch (Exception e) {
			Loger.log(TAG,
					"runnable worker error : " + e.getMessage());
			stopWork();
			taskListener.onFailure(new WorkerThreadException(e.getMessage()));
		} finally {
			if (conn != null) {
				conn.disconnect();
			}
		}

	}

	private void downSingle(RandomAccessFile taskRandomFile,
			RandomAccessFile downRandomFile, Task task, long length,
			long start, long end) {
		HttpURLConnection conn = null;
		try {
			// 这里我用HttpURLConnection下载，你也可以用HttpClient或者自己实现一个Http协议（不过貌似没有必要）

			URL u = new URL(task.getDownURL());

			conn = (HttpURLConnection) u.openConnection();
			initHeaders(conn);

			if (start > 0) {
				String range = "bytes=" + start + "-" + (end - 1);
				conn.setRequestProperty("Range", range);
			}
			conn.connect();
			if (start > 0 && conn.getContentLength() != (end - start)) {
				throw new RuntimeException(
						"conn.getContentLength() != (end - start)");
			}

			InputStream is = conn.getInputStream();

			byte[] temp = new byte[task.getBufferSize()];

			BufferedInputStream bis = new BufferedInputStream(is, temp.length);

			int readed = 0;
			long offset = task.getDownloadedLength();
			while ((readed = bis.read(temp)) > 0 && keepWorking) {
				downRandomFile.seek(offset);
				downRandomFile.write(temp, 0, readed);
				offset += readed;
				task.updateDownloadedLength(readed);
				task.writeOffset(taskRandomFile);
				if (!isPaused() && !isStoped()) {
					if (useDelay) {//采用延时策略,状态变化累积到1%才发通知~
						int percent = (int) ((double)task.getDownloadedLength()/(double)length*100);
						if (checkPercentChange(percent)) {
							taskListener.onUpdateProgress(readed,
									task.getDownloadedLength(), length);
						}
					} else {
						taskListener.onUpdateProgress(readed,
								task.getDownloadedLength(), length);
					}
				}
				if (needWait) {
					synchronized (task) {
						task.wait();
					}
				}
			}
			if (keepWorking) {
				success();
			}
		} catch (Exception e) {
			e.printStackTrace();
			Loger.log(TAG, "downSingle Error:" + e.getMessage());
			stopWork();
			taskListener.onFailure(new WorkerThreadException(e.getMessage()));
		} finally {
			if (conn != null) {
				conn.disconnect();
			}
		}
	}
	
	private boolean checkPercentChange(int percent) {
		if (percent - currnetPercent >= PERCENT_UNIT) {
			currnetPercent = percent;
			return true;
		} else {
			return false;
		}
	}

	private void initHeaders(HttpURLConnection conn) {
		conn.setRequestProperty("Connection", "Keep-Alive");
		conn.setRequestProperty("Charset", "UTF-8");
		conn.setRequestProperty("Accept-Language", "zh-CN");
		conn.setRequestProperty("User-Agent", "Android");
	}

	private void triggerSuccess(int sectionNo) {
		successTag[sectionNo] = true;
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
			success();
		}
	}

	public void stopWork() {
		keepWorking = false;
		checkRandomFileClose();
	}
	
	private void removeTaskFile() {
		// 任务描述文件
		if (taskFile == null && mTask != null) {
			taskFile = new File(mTask.getSaveFile() + APPEND_TASKFILE);
		}
		// 真正下载的文件
		if (saveFile == null && mTask != null) {
			saveFile = new File(mTask.getSaveFile() + APPEND_SAVEFIlE);
		}
		taskFile.delete();
		saveFile.delete();
	}

	public void reStartWork() {
		//force call to check
		stopWork();
		if (mTask != null) {
			keepWorking = true;
			work(mTask);
		}
	}
	
	public void pauseWork() {
		needWait = true;
		taskListener.pauseTask();
	}
	
	public void resumeWork() {
		needWait = false;
		synchronized (mTask) {
			mTask.notifyAll();
		}
		taskListener.resumeTask();
	}
	
	public boolean isPaused() {
		return needWait;
	}

	public boolean isStoped() {
		return !keepWorking;
	}
	
	/**
	 * stop working and delete file
	 */
	public void cancel() {
		pauseWork();
		stopWork();
		removeTaskFile();
	}
	
	private void initTaskFile(RandomAccessFile taskRandomFile, Task task)
			throws CreateTaskFileException {
		try {
			int secCount = task.getSectionCount();
			long per = task.getContentLength() / secCount;
			long[] sectionsOffset = new long[secCount];
			for (int i = 0; i < secCount; i++) {
				sectionsOffset[i] = per * i;
			}
			task.setSectionsOffset(sectionsOffset);
			task.create(taskRandomFile);
		} catch (Exception e) {
			throw new CreateTaskFileException(e.getMessage());
		}
	}

	public void setTaskListener(TaskListener taskListener) {
		this.taskListener = taskListener;
	}
}