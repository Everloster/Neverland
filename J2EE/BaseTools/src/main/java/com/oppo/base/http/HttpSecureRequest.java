/**
 * HttpSecureRequest.java
 * com.oppo.base.http
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-9-15 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.http;

import java.util.HashMap;
import java.util.Map;

import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;
import com.oppo.base.security.Hmac;

/**
 * ClassName:HttpSecureRequest <br>
 * Function: 经过签名的http连接 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-15  下午05:29:35
 */
public class HttpSecureRequest {
	private final static String METHOD_POST = "POST";
	private final static String METHOD_GET = "GET";
	
	/**
	 * 签名字段
	 */
	public final static String DEFAULT_HEADER_SIGNATURE = "nearme-internal-signature";
	
	/**
	 * key字段
	 */
	public final static String DEFAULT_HEADER_KEY = "nearme-internal-key";
	
	/**
	 * 时间戳
	 */
	public final static String DEFAULT_HEADER_TIMESTAMP = "nearme_internal_stamp";
	
	private String applicationKey;
	private String applicationSecurity;
	private String headerSignature = DEFAULT_HEADER_SIGNATURE;
	private String headerHeaderKey = DEFAULT_HEADER_KEY;
	private String headerTimestamp = DEFAULT_HEADER_TIMESTAMP;
	
	/**
	 * 设置应用的key和密钥
	 * @param 
	 * @return
	 */
	public void setSecureParam(String inputAppKey, String inputAppSecurity) {
		setSecureParam(inputAppKey, inputAppSecurity, DEFAULT_HEADER_SIGNATURE, DEFAULT_HEADER_KEY, DEFAULT_HEADER_TIMESTAMP);
	}
	
	/**
	 * 设置应用的key和密钥
	 * @param 
	 * @return
	 */
	public void setSecureParam(String inputAppKey, String inputAppSecurity,
			String headerSignatureName, String headerHeaderKeyName, String headerTimestampName) {
		applicationKey = inputAppKey;
		applicationSecurity = inputAppSecurity;
		headerSignature = headerSignatureName;
		headerHeaderKey = headerHeaderKeyName;
		headerTimestamp = headerTimestampName;
	}
	
	/**
	 * 通过POST方式获取数据
	 * @param 
	 * @return
	 */
	public byte[] getResponseByPost(String url, String requestString) throws Exception {
		return getResponseData(url, null, requestString, METHOD_POST);
	}
	
	/**
	 * 通过GET方式获取数据
	 * @param 
	 * @return
	 */
	public byte[] getResponseByGet(String url) throws Exception {
		return getResponseData(url, null, null, METHOD_GET);
	}
	
	/**
	 * 通过指定链接和参数获取返回结果
	 * @param 
	 * @return
	 */
	public byte[] getResponseData(String urlLink, Map<String, String> headers,
			String requestString, String method) throws Exception {
		if(!checkSecureSetting()) {
			throw new Exception("请调用setSecureParam设置applicationKey和applicationSecurity");
		}
		
		//重设header
		if(null == headers) {
			headers = new HashMap<String, String>(3);
		}
		
		//得到需要签名的内容
		String signString = null;
		if(METHOD_GET.equals(method)) { //get方法对url的query string签名
			int index = urlLink.indexOf('?');
			if(index > 0) {
				signString = urlLink.substring(index + 1);
			}
		} else {	//post方法对传递的数据进行签名
			signString = requestString;
		}
		
		if(null == signString) {
			signString = OConstants.EMPTY_STRING;
		}
		
		//计算签名值
		String sign;
		if(StringUtil.isNullOrEmpty(headerTimestamp)) {
			sign = Hmac.signSHA1(signString, applicationSecurity);
		} else {
			String now = String.valueOf(System.currentTimeMillis());
			sign = Hmac.signSHA1(signString + now, applicationSecurity);
			headers.put(headerTimestamp, now);
		}

		//设置签名header
		headers.put(headerSignature, sign);
		headers.put(headerHeaderKey, applicationKey);

		return HttpDownload.getResponseData(urlLink, headers, requestString, method);
	}
	
	/**
	 * 检查applicationKey和applicationSecurity是否设置正确
	 * @param 
	 * @return
	 */
	private boolean checkSecureSetting() {
		return !StringUtil.isNullOrEmpty(applicationKey)
			&& !StringUtil.isNullOrEmpty(applicationSecurity);
	}
}

