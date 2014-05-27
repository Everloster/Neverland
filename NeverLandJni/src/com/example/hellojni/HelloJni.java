/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.hellojni;

import java.io.File;
import java.io.IOException;

import com.nearme.gamecenter.jnitool.JniTools;

import android.app.Activity;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.TextView;
import android.widget.Toast;
import android.os.Bundle;
import android.os.Environment;

public class HelloJni extends Activity {
	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		/*
		 * Create a TextView and set its content. the text is retrieved by
		 * calling a native function.
		 */

		final PackageManager pm = this.getPackageManager();
		PackageInfo pi = null;
		try {
			pi = pm.getPackageInfo(this.getPackageName(), 0);
		} catch (NameNotFoundException e) {
			e.printStackTrace();
		}
		final String path = pi.applicationInfo.sourceDir;
		final TextView tv = new TextView(this);
		tv.setText(getData());
		setContentView(tv);

		tv.setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {
				test(path, tv);
			}
		});
		tv.setClickable(true);
	}

	/*
	 * A native method that is implemented by the 'hello-jni' native library,
	 * which is packaged with this application.
	 */
	public native String stringFromJNI();

	/*
	 * This is another native method declaration that is *not* implemented by
	 * 'hello-jni'. This is simply to show that you can declare as many native
	 * methods in your Java code as you want, their implementation is searched
	 * in the currently loaded native libraries only the first time you call
	 * them.
	 * 
	 * Trying to call this function will result in a
	 * java.lang.UnsatisfiedLinkError exception !
	 */
	public native String unimplementedStringFromJNI();

	/*
	 * this is used to load the 'hello-jni' library on application startup. The
	 * library has already been unpacked into
	 * /data/data/com.example.hellojni/lib/libhello-jni.so at installation time
	 * by the package manager.
	 */

	public native String getData();

	private void test(final String path, final TextView tv) {
		// tv.setText(JniTools.getFileMD5(path).equals(HashUtil.md5File(new
		// File(path))) + "");
		final File oldApk = new File(Environment.getExternalStorageDirectory(),
				"gc1_1.apk");
		final File newApk = new File(Environment.getExternalStorageDirectory(),
				"gc1_2.apk");
		final File tmp = new File(Environment.getExternalStorageDirectory(),
				"temp.path");
		final File patchedNewApk = new File(
				Environment.getExternalStorageDirectory(), "gc1_2_new.apk");
		if (tmp.exists()) {
			tmp.delete();
		}
		if (patchedNewApk.exists()) {
			patchedNewApk.delete();
		}
		new Thread(new Runnable() {

			@Override
			public void run() {
				JniTools.genDiff(oldApk.getAbsolutePath(),
						newApk.getAbsolutePath(), tmp.getAbsolutePath());
				System.out.println("diff ok");
				JniTools.patch(oldApk.getAbsolutePath(),
						patchedNewApk.getAbsolutePath(), tmp.getAbsolutePath());
				System.out.println("diff ok");
				if (JniTools.getFileMD5(newApk.getAbsolutePath())
						.equals(JniTools.getFileMD5(patchedNewApk.getAbsolutePath()))) {
//					Toast.makeText(getApplicationContext(), "ok", Toast.LENGTH_LONG).show();
					System.out.println("md5 the same ok");
				} else {
					System.out.println("md5 the same failure");
				}
			}
		}).start();

	}

	static {
		System.loadLibrary("GCJniTool");
	}
}
