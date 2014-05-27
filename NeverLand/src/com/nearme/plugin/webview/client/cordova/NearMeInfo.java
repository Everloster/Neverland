/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2013-9-11
 */
package com.nearme.plugin.webview.client.cordova;

import org.apache.cordova.CallbackContext;
import org.apache.cordova.CordovaPlugin;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.nearme.webview.plugin.PluginConfig;

/**
 * 
 * @Author	lailong
 * @Since	2013-9-11
 */
public class NearMeInfo extends CordovaPlugin{
	
	/* (non-Javadoc)
	 * @see org.apache.cordova.CordovaPlugin#execute(java.lang.String, org.json.JSONArray, org.apache.cordova.CallbackContext)
	 */
	@Override
	public boolean execute(String action, JSONArray args,
			CallbackContext callbackContext) throws JSONException {
		try {
			if (action.equalsIgnoreCase("getNearMeInfo")) {
				final JSONObject jsonObject = new JSONObject();
				jsonObject.put("productId", PluginConfig.sProductId);
				jsonObject.put("appKey", PluginConfig.sAppKey);
				jsonObject.put("tokenKey", PluginConfig.sTokenKey);
				callbackContext.success(jsonObject);
			} else {
				return false;
			} 
			return true;
		} catch (Exception e) {
			return false;
		}
	}
}
