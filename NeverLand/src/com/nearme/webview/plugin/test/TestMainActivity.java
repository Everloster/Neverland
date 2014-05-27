/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2013-9-11
 */
package com.nearme.webview.plugin.test;

import org.jabe.neverland.R;

import android.app.Activity;
import android.app.ProgressDialog;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;

import com.nearme.webview.plugin.PluginConfig;
import com.nearme.webview.plugin.PluginLoader;
import com.nearme.webview.plugin.PluginLoaderInterface;

/**
 * 
 * @Author lailong
 * @Since 2013-9-11
 */
public class TestMainActivity extends Activity {

	private PluginLoader pluginLoader;

	/*
	 * (non-Javadoc)
	 * 
	 * @see android.app.Activity#onCreate(android.os.Bundle)
	 */
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		setContentView(R.layout.test_plugin_main);

		findViewById(R.id.button).setOnClickListener(new OnClickListener() {

			@Override
			public void onClick(View v) {

				final ProgressDialog dialog = new ProgressDialog(
						TestMainActivity.this);
				dialog.setMessage("加载中。。。");
				final PluginConfig config = new PluginConfig.Builder()
						.addAppKey("c5217trjnrmU6gO5jG8VvUFU0")
						.addProductId("10011001")
						.addTokenKey("b1f430588d648160b7a49f16e9aebc42")
						.build();
				pluginLoader = new PluginLoader(TestMainActivity.this, config,
						new PluginLoaderInterface() {

							@Override
							public void onBeforeBackground() {
								// TODO Auto-generated method stub
								dialog.show();
							}

							@Override
							public void onBackgroundFinish() {
								// TODO Auto-generated method stub
								dialog.dismiss();
							}

							@Override
							public void onLoadFinish(Result result) {
								// TODO Auto-generated method stub
								pluginLoader.openPlugin();
							}

						});
				pluginLoader.reLoadPluginFile();
				pluginLoader.startLoading();
			}
		});
	}
}
