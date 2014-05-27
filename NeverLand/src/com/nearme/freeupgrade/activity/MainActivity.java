package com.nearme.freeupgrade.activity;

import java.io.File;

import org.jabe.neverland.R;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;

import com.nearme.freeupgrade.lib.PluginProxyActivity;
import com.nearme.freeupgrade.lib.pinterface.PluginInterface;
import com.nearme.freeupgrade.lib.util.Constants;
import com.nearme.freeupgrade.util.FileHandler;
import com.nearme.freeupgrade.util.Util;

public class MainActivity extends Activity implements OnClickListener {
	
	private final static String TAG = MainActivity.class.getSimpleName();
	
	Button btnLoadApk;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main_free_upgrade);
		checkDir();
		copyNewPlugin();
		initViews();
		PluginInterface.initPluginConfig(new File(Constants.ROOT_DIR_PATH));
		PluginInterface.checkPlugins(this, null);
	}

	void initViews() {
		btnLoadApk = (Button) findViewById(R.id.btn_loadapk);
		btnLoadApk.setOnClickListener(this);
		findViewById(R.id.btn_loadwidget).setOnClickListener(this);
	}

	void checkDir() {
		FileHandler.getInstance().makeSureDirectoryExists(
				Environment.getExternalStorageDirectory() + File.separator
						+ Constants.ROOT_DIR_NAME);
		FileHandler.getInstance().makeSureDirectoryExists(
				Environment.getExternalStorageDirectory() + File.separator
						+ Constants.ROOT_DIR_NAME + "/.plugin");
	}

	void copyNewPlugin() {
		Util.CopyAssertTOSD(this, Constants.ROOT_DIR_PATH + File.separator + ".plugin",
				Constants.PLUGIN_FILE_NAME);
	}

	@Override
	public void onClick(View v) {
		switch (v.getId()) {
			case R.id.btn_loadapk:
				final Intent intent = new Intent();
				intent.setClass(this, PluginProxyActivity.class);
				intent.putExtra(PluginProxyActivity.EXTRA_PLUGIN_APKPATH,
						Constants.PLUGIN_FILE_PATH);
				intent.putExtra(PluginProxyActivity.EXTRA_LAUNCH_ACTIVITY, "com.nearme.plugin.webview.WebActivity");
				intent.putExtra("url","http://gamecenter.wanyol.com/gcbbs/forum.php?token=fbb9aa19e9cee99d2298901fc00837b3&secret=5cedd1bed79f5c10b8c8109003e2f7fd&appKey=c5217trjnrmU6gO5jG8VvUFU0");
				startActivity(intent);
				break;
			case R.id.btn_loadwidget: {
				final Intent intent1 = new Intent(this, ViewInflageActivity.class);
				startActivity(intent1);
				break;
			}
			default:
				break;
		}
	}

}
