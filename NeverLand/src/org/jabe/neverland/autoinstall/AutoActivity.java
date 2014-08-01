/**
 * Copyright (C) 2014, all rights reserved.
 * Company	Alibaba Group. 
 * Author	LaiLong
 * Since	Jul 29, 2014
 */
package org.jabe.neverland.autoinstall;

import java.io.DataOutputStream;
import java.io.File;
import java.util.List;
import java.util.concurrent.Executors;

import org.jabe.neverland.autoinstall.AutoInstallHelper.InstallCallback;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.LinearLayout.LayoutParams;
import android.widget.TextView;

import com.stericson.RootTools.RootTools;

/**
 * 
 * @Author LaiLong
 * @Since Jul 29, 2014
 */
public class AutoActivity extends Activity {

	private TextView mDesription;
	private Button mButton;
	private Button mUninstallButton;

	/*
	 * (non-Javadoc)
	 * 
	 * @see android.app.Activity#onCreate(android.os.Bundle)
	 */
	@Override
	protected void onCreate(Bundle savedInstanceState) {

		super.onCreate(savedInstanceState);

		final LinearLayout content = new LinearLayout(this);
		content.setOrientation(LinearLayout.VERTICAL);

		setContentView(content);

		mDesription = new TextView(this);
		mDesription.setTextColor(Color.WHITE);
		mDesription.setTextSize(20);
		mDesription.setText("This is a test for auto install/uninstall an apk file.");

		mButton = new Button(this);

		mButton.setText("TryAutoInstall");
		final LinearLayout.LayoutParams params = new LayoutParams(LayoutParams.WRAP_CONTENT,
				LayoutParams.WRAP_CONTENT);
		mButton.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {
				if (v == mButton) {
					if (RootTools.isRootAvailable()) {
						if (upgradeRootPermission(getPackageCodePath())) {
							tryRandomInstall();
						} else {
							mDesription.setText("request root failure!");
						}
					} else {
						mDesription.setText("device is not root!");
					}

				}

			}

		});
		
		mUninstallButton = new Button(this);
		mUninstallButton.setText("TryUninstall");
		mUninstallButton.setOnClickListener(new OnClickListener() {
			
			@Override
			public void onClick(View v) {
				if (v == mUninstallButton) {
					if (RootTools.isRootAvailable()) {
						if (upgradeRootPermission(getPackageCodePath())) {
							tryRandomUninstall();
						} else {
							mDesription.setText("request root failure!");
						}
					} else {
						mDesription.setText("device is not root!");
					}
				}
			}
		});

		content.addView(mButton, params);
		content.addView(mUninstallButton, params);
		content.addView(mDesription, params);
	}

	private boolean useHelper = false;

	private void tryRandomInstall() {
		final ProgressDialog dialog = ProgressDialog.show(AutoActivity.this, "加载中", "Loading");
		final PackageManager pm = AutoActivity.this.getPackageManager();
		List<PackageInfo> packages = pm.getInstalledPackages(0);
		for (PackageInfo packageInfo : packages) {
			final PackageInfo pkg = packageInfo;
			if ((packageInfo.applicationInfo.flags & ApplicationInfo.FLAG_SYSTEM) == 0) {

				if (useHelper) {
					useHelpr(dialog, packageInfo, pkg);
				} else {
					usePackageUtil(dialog, pkg);
				}

				break;
			}
		}
	}
	
	private void tryRandomUninstall() {
		final ProgressDialog dialog = ProgressDialog.show(AutoActivity.this, "加载中", "Loading");
		final PackageManager pm = AutoActivity.this.getPackageManager();
		List<PackageInfo> packages = pm.getInstalledPackages(0);
		for (PackageInfo packageInfo : packages) {
			final PackageInfo pkg = packageInfo;
			if ((packageInfo.applicationInfo.flags & ApplicationInfo.FLAG_SYSTEM) == 0) {

				final Handler handler = new Handler(Looper.getMainLooper()) {

					@Override
					public void dispatchMessage(Message msg) {
						dialog.dismiss();
						mDesription.setText("Uninstall Result :" + (msg.what == 1 ? "success" : msg.what)
								+ " [" + pkg.packageName + "]");
						mDesription.append(" message [" + msg.what + "]");
					}
				};
				
				final Runnable r = new Runnable() {

					@Override
					public void run() {
						int result = PackageUtils.uninstallSilent(AutoActivity.this, pkg.packageName);
						final Message msg = handler.obtainMessage();
						msg.what = result;
						msg.sendToTarget();
					}
				};
				
				new Thread(r).start();

				break;
			}
		}
	}

	private void usePackageUtil(final ProgressDialog dialog, final PackageInfo pkg) {
		final Handler handler = new Handler(Looper.getMainLooper()) {

			@Override
			public void dispatchMessage(Message msg) {
				dialog.dismiss();
				mDesription.setText("Install Result :" + (msg.what == 1 ? "success" : msg.what)
						+ " [" + pkg.packageName + "]");
				mDesription.append(" message ["
						+ AutoInstallHelper.getInstallResultByCode(msg.what) + "]");
			}
		};
		final Runnable r = new Runnable() {

			@Override
			public void run() {
				int result = PackageUtils.installSilent(AutoActivity.this,
						pkg.applicationInfo.sourceDir);
				final Message msg = handler.obtainMessage();
				msg.what = result;
				msg.sendToTarget();
			}
		};
		new Thread(r).start();
	}

	private void useHelpr(final ProgressDialog dialog, PackageInfo packageInfo,
			final PackageInfo pkg) {
		AutoInstallHelper.autoInstallAsync(new File(packageInfo.applicationInfo.sourceDir),
				new InstallCallback() {

					@Override
					public void onSuccess() {
						mDesription.setText("Install Success.");
						dialog.dismiss();
					}

					@Override
					public void onFailure(int code, String message) {
						mDesription.setText("Install failure : ");
						mDesription.append(" file[" + pkg.packageName + "] ");
						mDesription.append(" messaeg[" + message + "].");
						mDesription.invalidate();
						dialog.dismiss();
					}

				}, Executors.newCachedThreadPool());
	}

	/**
	 * 应用程序运行命令获取 Root权限，设备必须已破解(获得ROOT权限)
	 * 
	 * @return 应用程序是/否获取Root权限
	 */
	public static boolean upgradeRootPermission(String pkgCodePath) {
		Process process = null;
		DataOutputStream os = null;
		try {
			String cmd = "chmod 777 " + pkgCodePath;
			process = Runtime.getRuntime().exec("su"); // 切换到root帐号
			os = new DataOutputStream(process.getOutputStream());
			os.writeBytes(cmd + "\n");
			os.writeBytes("exit\n");
			os.flush();
			process.waitFor();
		} catch (Exception e) {
			return false;
		} finally {
			try {
				if (os != null) {
					os.close();
				}
				process.destroy();
			} catch (Exception e) {
			}
		}
		return true;
	}
}
