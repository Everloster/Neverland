package org.jabe.neverland.activity;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;

import org.jabe.neverland.R;
import org.jabe.neverland.download.Task;
import org.jabe.neverland.download.TaskAssign;
import org.jabe.neverland.download.TaskListener;
import org.jabe.neverland.util.Util;

import android.app.Activity;
import android.app.ProgressDialog;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.os.StatFs;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.Toast;

public class DownloadActivity extends Activity {

	private Button mPauseButton, mResumeButton, mDeleteButton, mRestartButton;
	private ProgressBar mProgressBar;
	private ProgressDialog mProgressDialog;
	private long contentLength = -1;
	private Task task;
	private TaskAssign taskAssign;
	private Handler mHandler = new Handler() {
		@Override
		public void dispatchMessage(android.os.Message msg) {
			Bundle b = msg.getData();
			double d = b.getDouble("downloaded");
			if (d > 0) {
				mProgressBar.setProgress((int) d);
			}
		};
	};

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.act_download_manager);
		setUpView();
//		testBigFile();
		makeToast("AvailaleSize SD size : " + getAvailaleSize()/1024/1024 + "M");
	}

	@SuppressWarnings("unused")
	private void testBigFile() {
		File file = new File(Environment.getExternalStorageDirectory()
				.getAbsolutePath() + File.separator + "a.temp");
		try {
			RandomAccessFile ran = new RandomAccessFile(file, "rw");
			// 存储空间不够抛出的是IOException, 文件会被创建, 大小为剩余容量, 所以创建失败后应该删除.
			ran.setLength(new Long(1024 * 1024 * 1024));

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public long getAvailaleSize() {

		File path = Environment.getExternalStorageDirectory(); // 取得sdcard文件路径

		StatFs stat = new StatFs(path.getPath());

		long blockSize = stat.getBlockSize();

		long availableBlocks = stat.getAvailableBlocks();

		return availableBlocks * blockSize;

		// (availableBlocks * blockSize)/1024 KIB 单位

		// (availableBlocks * blockSize)/1024 /1024 MIB单位

	}

	@SuppressWarnings("unused")
	private void beginToDown() {
		final String url1 = "http://storefs.nearme.com.cn/uploadFiles/Pfiles/201205/104ca8a227e1429ca638f3450217dff5.apk";
		final String url2 = "http://go.microsoft.com/fwlink/?linkid=57034";
		final String name1 = "水果连连看.apk";
		final String name2 = "vc2005express.iso";
		task = new Task();
		task.setDownURL(url1);
		task.setSaveFile(Environment.getExternalStorageDirectory()
				.getAbsolutePath() + File.separator + name1);
		task.setSectionCount(1);
		task.setWorkerCount(1);
		task.setBufferSize(8 * 1024);
		taskAssign = new TaskAssign();
		taskAssign.setTaskListener(new TaskListener("") {
			@Override
			public void resumeTask() {
				makeToast("Resume to download");
			}

			@Override
			public void onUpdateProgress(double added, double downloaded,
					double total) {
				Message message = mHandler.obtainMessage();
				Bundle b = new Bundle();
				b.putDouble("downloaded", downloaded);
				b.putDouble("total", total);
				message.setData(b);
				message.sendToTarget();
			}

			@Override
			public void onSuccess() {
				makeToast("Finish download");
			}

			@Override
			public void onPreTask() {
				makeToast("Begin to download");
			}

			@Override
			public void onFailure(Exception e) {
				makeToast("Download failure");
			}
			@Override
			public void onFileExist(File file) {
				makeToast("File Exist");
			}
		});

		AsyncTask<Void, Void, Long> aTask = new AsyncTask<Void, Void, Long>() {
			@Override
			protected void onPreExecute() {
				mProgressDialog.show();
			}

			@Override
			protected Long doInBackground(Void... params) {
				try {
					return Task.getContentLength(task.getDownURL());
				} catch (IOException e) {
					makeToast("Net Error");
					return new Long(0);
				}
			}

			@Override
			protected void onPostExecute(Long result) {
				mProgressDialog.dismiss();
				contentLength = result;
				if (contentLength < 0) {
					return;
				}
				mProgressBar.setMax((int) contentLength);
				mProgressBar.setProgress(0);
				task.setContentLength(contentLength);
				taskAssign.work(task);

			}
		};
		aTask.execute();
		mPauseButton.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {
				taskAssign.pauseWork();
			}
		});
		mResumeButton.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {
				if (taskAssign != null) {
					taskAssign.resumeWork();
				} else {
					beginToDown();
				}
			}
		});
		mRestartButton.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {
				mProgressBar.setProgress(0);
				taskAssign.reStartWork();
			}
		});
		mDeleteButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				File file = new File(task.getSaveFile());
				if(file.delete()) {
					Util.showToast(DownloadActivity.this, "删除文件成功!");
				}
			}
		});

	}

	private void setUpView() {
		mPauseButton = (Button) findViewById(R.id.button1);
		mResumeButton = (Button) findViewById(R.id.button2);
		mDeleteButton = (Button) findViewById(R.id.button3);
		mRestartButton = (Button) findViewById(R.id.button4);
		mProgressBar = (ProgressBar) findViewById(R.id.progressBar1);
		mProgressDialog = new ProgressDialog(this);
		mProgressDialog.setCancelable(false);
		mProgressDialog.setTitle("Download Initial...");
	}

	public void makeToast(final String content) {
		this.runOnUiThread(new Runnable() {
			@Override
			public void run() {
				Toast.makeText(DownloadActivity.this, content, 1000).show();
			}
		});
	}
}
