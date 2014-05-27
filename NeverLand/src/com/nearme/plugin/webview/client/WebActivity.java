/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2013-9-2
 */
package com.nearme.plugin.webview.client;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.cordova.Config;
import org.apache.cordova.CordovaChromeClient;
import org.apache.cordova.CordovaInterface;
import org.apache.cordova.CordovaPlugin;
import org.apache.cordova.CordovaWebView;
import org.jabe.neverland.R;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.KeyEvent;
import android.widget.LinearLayout;
import android.widget.LinearLayout.LayoutParams;

import com.nearme.webview.plugin.impl.PluginImpl;


/**
 * 
 * @Author	lailong
 * @Since	2013-9-2
 */
public class WebActivity extends PluginImpl implements CordovaInterface {
	
	private CordovaWebView mCordovaWebView;
	
	private CordovaChromeClient mCordovaChromeClient;
	
	private final ExecutorService mThreadPool = Executors.newCachedThreadPool();
	
	/* (non-Javadoc)
	 * @see android.app.Activity#onCreate(android.os.Bundle)
	 */
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		
		Config.init(this);
		
		setContentView(R.layout.test_plugin_client);
        
        addWebView();
        
        initClient();
	}

	/**
	 * 
	 */
	private void initClient() {
		mCordovaChromeClient = mCordovaWebView.getWebChromeClient();
	}

	private void addWebView() {
		
		final LinearLayout root = (LinearLayout) findViewById(R.id.root);

        mCordovaWebView = new CordovaWebView(this, null);
        
        final LinearLayout.LayoutParams layoutParams = new LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT);
        
        root.addView(mCordovaWebView, layoutParams);

        mCordovaWebView.loadUrl("file:///android_asset/www/index.html");
        
//        mCordovaWebView.loadUrl("http://www.baidu.com");
        
	}

	/* (non-Javadoc)
	 * @see com.nearme.webview.plugin.IPlugin#onPluginActivityResult(int, int, android.content.Intent)
	 */
	@Override
	public void onPluginActivityResult(int requestCode, int resultCode, Intent data) {
		
		if (requestCode == CordovaChromeClient.FILECHOOSER_RESULTCODE) {
			Uri result = data == null || resultCode != RESULT_OK ? null
					: data.getData();
			mCordovaChromeClient.getValueCallback().onReceiveValue(result);
		}
		
		CordovaPlugin callback = this.activityResultCallback;
        
        if(callback != null) {
            callback.onActivityResult(requestCode, resultCode, data);
        }
		
	}

	/* (non-Javadoc)
	 * @see com.nearme.webview.plugin.IPlugin#onPluginDestroy()
	 */
	@Override
	public void onPluginDestroy() {
		if (mCordovaWebView != null) {
			mCordovaWebView.handleDestroy();
		}
	}

	/* (non-Javadoc)
	 * @see com.nearme.webview.plugin.IPlugin#onPluginKeyDown(int, android.view.KeyEvent)
	 */
	@Override
	public boolean onPluginKeyDown(int arg0, KeyEvent arg1) {
		// TODO Auto-generated method stub
		return false;
	}

	/* (non-Javadoc)
	 * @see com.nearme.webview.plugin.IPlugin#onPluginPause()
	 */
	@Override
	public void onPluginPause() {
		// TODO Auto-generated method stub
		
	}

	/* (non-Javadoc)
	 * @see com.nearme.webview.plugin.IPlugin#onPluginResume()
	 */
	@Override
	public void onPluginResume() {
		// TODO Auto-generated method stub
		
	}

	/* (non-Javadoc)
	 * @see com.nearme.webview.plugin.IPlugin#onPluginStart()
	 */
	@Override
	public void onPluginStart() {
		// TODO Auto-generated method stub
		
	}

	/* (non-Javadoc)
	 * @see com.nearme.webview.plugin.IPlugin#onPluginStop()
	 */
	@Override
	public void onPluginStop() {
		// TODO Auto-generated method stub
		
	}
	
	
	// Plugin to call when activity result is received
    protected CordovaPlugin activityResultCallback = null;
    protected boolean activityResultKeepRunning;
    protected boolean keepRunning = true;

	/* (non-Javadoc)
	 * @see org.apache.cordova.CordovaInterface#startActivityForResult(org.apache.cordova.CordovaPlugin, android.content.Intent, int)
	 */
	@Override
	public void startActivityForResult(CordovaPlugin command, Intent intent,
			int requestCode) {
		
		this.activityResultCallback = command;
        this.activityResultKeepRunning = this.keepRunning;

        // If multitasking turned on, then disable it for activities that return results
        if (command != null) {
            this.keepRunning = false;
        }
        
        // Start activity
        super.startActivityForResult(intent, requestCode);
		
	}

	/* (non-Javadoc)
	 * @see org.apache.cordova.CordovaInterface#setActivityResultCallback(org.apache.cordova.CordovaPlugin)
	 */
	@Override
	public void setActivityResultCallback(CordovaPlugin plugin) {
		// TODO Auto-generated method stub
		this.activityResultCallback = plugin;
	}

	/* (non-Javadoc)
	 * @see org.apache.cordova.CordovaInterface#getActivity()
	 */
	@Override
	public Activity getActivity() {
		// TODO Auto-generated method stub
		return this;
	}

	/* (non-Javadoc)
	 * @see org.apache.cordova.CordovaInterface#onMessage(java.lang.String, java.lang.Object)
	 */
	@Override
	public Object onMessage(String id, Object data) {
		if ("exit".equals(id)) {
            this.finish();
        }
		return null;
	}

	/* (non-Javadoc)
	 * @see org.apache.cordova.CordovaInterface#getThreadPool()
	 */
	@Override
	public ExecutorService getThreadPool() {
		// TODO Auto-generated method stub
		return mThreadPool;
	}
	
	
	
}
