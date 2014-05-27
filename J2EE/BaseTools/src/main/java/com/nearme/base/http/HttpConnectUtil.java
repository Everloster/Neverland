/**
 * HttpConnectUtil.java
 * com.nearme.base.http
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-8-11 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.http;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpUriRequest;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.message.BasicNameValuePair;

import com.oppo.base.common.OConstants;
import com.oppo.base.file.FileOperate;

/**
 * ClassName:HttpConnectUtil <br>
 * Function: http连接工具类（连接池） <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-8-11  上午11:53:25
 */
public class HttpConnectUtil {
	private static final int HTTP_STATUS_OK = 200;
	
	/**
	 * 使用get方式请求http数据
	 * @param url 请求的url
	 * @param headers 请求时的头信息
	 * @return
	 */
	public static byte[] getResponseByGet(String url, Map<String, String> headers) throws Exception {
		return getResponseByGet(null, url, headers);
	}
	
	/**
	 * 使用get方式请求http数据
	 * 
	 * @param client
	 * @param url 请求的url
	 * @param headers 请求时的头信息
	 * @return
	 */
	public static byte[] getResponseByGet(HttpClient client, String url, Map<String, String> headers) throws Exception {
		HttpGet get = new HttpGet(url);
		if(null != headers) {
			for(String key : headers.keySet()) {
				get.addHeader(key, headers.get(key));
			}
		}
		
		return getResponseData(client, get);
	}
	
	/**
	 * 使用post方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param requestString 请求数据
	 * @return
	 */
	public static byte[] getResponseByPost(String urlLink, Map<String, String> headers, String requestString) throws Exception {
		return getResponseByPost(null, urlLink, headers, requestString);
	}
	
	/**
	 * 使用post方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param requestString 请求数据
	 * @return
	 */
	public static byte[] getResponseByPost(HttpClient client, String urlLink, Map<String, String> headers, String requestString) throws Exception {
		byte[] data = null;
		//发送数据
		if(null != requestString) {
			data = requestString.getBytes(OConstants.DEFAULT_ENCODING);	
		}
		
		return getResponseByPost(client, urlLink, headers, data);
	}
	
	/**
	 * 使用post方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param paramMap 请求参数
	 * @return
	 */
	public static byte[] getResponseByPost(String urlLink, Map<String, String> headers, Map<String, String> paramMap) throws Exception {
		return getResponseByPost(null, urlLink, headers, paramMap);
	}
	
	/**
	 * 使用post方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param paramMap 请求参数
	 * @return
	 */
	public static byte[] getResponseByPost(HttpClient client, String urlLink, Map<String, String> headers, Map<String, String> paramMap) throws Exception {
		HttpPost post = new HttpPost(new URI(urlLink));
		if(null != headers) {
			for(String key : headers.keySet()) {
				post.addHeader(key, headers.get(key));
			}
		}
		
		//发送数据
		if(null != paramMap) {
			List<BasicNameValuePair> paramList = new ArrayList<BasicNameValuePair>(paramMap.size());
			for(String key : paramMap.keySet()) {
				paramList.add(new BasicNameValuePair(key, paramMap.get(key)));
			}
			
			post.setEntity(new UrlEncodedFormEntity(paramList, OConstants.DEFAULT_ENCODING));
		}
		
		return getResponseData(client, post);
	}
	
	/**
	 * 使用指定的方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param requestString 请求数据
	 * @return
	 */
	public static byte[] getResponseByPost(String urlLink, Map<String, String> headers, byte[] requestBytes) throws Exception {
		return getResponseByPost(null, urlLink, headers, requestBytes);
	}
	
	/**
	 * 使用指定的方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param requestString 请求数据
	 * @return
	 */
	public static byte[] getResponseByPost(HttpClient client, String urlLink, Map<String, String> headers, byte[] requestBytes) throws Exception {
		HttpPost post = new HttpPost(new URI(urlLink));
		if(null != headers) {
			for(String key : headers.keySet()) {
				post.addHeader(key, headers.get(key));
			}
		}
		post.setEntity(new ByteArrayEntity(requestBytes));
		
		return getResponseData(client, post);
	}
	
	/**
	 * 根据指定参数获取http数据
	 * @param 
	 * @return
	 */
	public static byte[] getResponseData(HttpClient client, HttpUriRequest request) throws Exception {
		if(null == client) {
			client = HttpConnectManager.getHttpClient();
		}
		
		InputStream is = null;
		try {
			HttpResponse response = client.execute(request);
			HttpEntity entity = response.getEntity();
			if (null != entity) {
				is = entity.getContent();
			}
			
			if(HTTP_STATUS_OK == response.getStatusLine().getStatusCode()) {
				if (null != is) {
					return getData(is, (int)entity.getContentLength());
				}
			} else {
				request.abort();
			}
		} catch(Exception ex) {
			request.abort();
			throw ex;
		} finally {
			FileOperate.close(is);
		}
		
		return null;
	}
	
	/**
	 * 获取url跳转后的链接
	 * @param 
	 * @return
	 */
	public static String getRedirectUrl(String urlLink, Map<String, String> headers) throws Exception {
		URL url = new URL(urlLink);
		HttpURLConnection conn = null;
		InputStream is = null;
		try {
			conn = (HttpURLConnection)url.openConnection();
			if(null != headers) {
				for(String key : headers.keySet()) {
					conn.setRequestProperty(key, headers.get(key));
				}
			}
			
			conn.connect();
			is = conn.getInputStream();
			return conn.getURL().toString();
		} catch(Exception ex) {
			throw ex;
		} finally {
			FileOperate.close(is);
			
			//关闭连接
			if(null != conn) {
				conn.disconnect();
			}
		}
	}
	
	public static byte[] getData(InputStream is, int len) throws Exception {
		if(len <= 0) {
			len = 32;
		}
		
		ByteArrayOutputStream baos = new ByteArrayOutputStream(len);
		int bufferLen = 4096;
		byte[] buff = new byte[bufferLen];
		int readLen;
		while ((readLen = is.read(buff, 0, bufferLen)) != -1) {
			baos.write(buff, 0, readLen);
		}
		
		return baos.toByteArray();
	}
	
	public static void main(String[] args) throws Exception {
		
	}
}

