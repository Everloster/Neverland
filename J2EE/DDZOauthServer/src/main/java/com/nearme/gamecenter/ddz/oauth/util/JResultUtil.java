/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年1月21日
 */
package com.nearme.gamecenter.ddz.oauth.util;

import org.json.JSONObject;

import com.nearme.gamecenter.ddz.oauth.handler.CoreProxyHandler;

/**
 * 
 * @Author	LaiLong
 * @Since	2014年1月21日
 */
public class JResultUtil {
	
	public static String getOkJsonString() {
		final JSONObject jsonObject = new JSONObject();
		jsonObject.put("resultCode", CoreProxyHandler.RESULT_OK);
		jsonObject.put("resultMsg", "OK");
		return jsonObject.toString();
	}
	
	public static String getOkJsonString(String msg) {
		final JSONObject jsonObject = new JSONObject();
		jsonObject.put("resultCode", CoreProxyHandler.RESULT_OK);
		jsonObject.put("resultMsg", msg);
		return jsonObject.toString();
	}

	public static String getFailureJsonString(int failureCode) {
		final JSONObject jsonObject = new JSONObject();
		jsonObject.put("resultCode", failureCode);
		jsonObject.put("resultMsg", "FAILURE");
		return jsonObject.toString();
	}
	
}
