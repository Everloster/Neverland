/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2013-9-11
 */
package com.nearme.webview.plugin;

/**
 * 
 * @Author	lailong
 * @Since	2013-9-11
 */
public interface PluginLoaderInterface {
	
	public void onBeforeBackground();
	
	public void onBackgroundFinish();
	
	public void onLoadFinish(Result result);
	
	public static enum Result {
		
		SUCCESS,
		NETWORK_ERROR,
		UNKNOWN_FAIL
		
	}
}
