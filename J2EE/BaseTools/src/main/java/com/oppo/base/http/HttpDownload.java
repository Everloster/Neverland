/**
 * HttpDownload.java
 * com.oppo.base.http
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-20 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.http;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.Map;

import com.oppo.base.common.OConstants;
import com.oppo.base.file.FileOperate;

/**
 * ClassName:HttpDownload
 * Function: http文件下载
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-7-20  上午10:18:34
 */
public class HttpDownload {
	/**
	 * 不传递header使用post方式获取http数据
	 * @param url post的url
	 * @param requestString 请求时的数据
	 * @return
	 */
	public static byte[] getResponseByPost(String url, String requestString) throws Exception {
		return getResponseData(url, null, requestString, "POST");
	}

	/**
	 * 使用get方式请求http数据
	 * @param url 请求的url
	 * @return
	 */
	public static byte[] getResponseByGet(String url) throws Exception {
		return getResponseData(url, null, null, "GET");
	}

	/**
	 * 使用指定的方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param requestString 请求数据
	 * @param method 请求方式，如果requestString不为null，则method将被处理为post
	 * @return
	 */
	public static byte[] getResponseData(String urlLink, Map<String, String> headers, String requestString, String method) throws Exception {
		byte[] data = null;
		//发送数据
		if(null != requestString) {
			data = requestString.getBytes(OConstants.DEFAULT_ENCODING);
		}

		return getResponseData0(urlLink, headers, data, method);
	}

	/**
	 * 使用指定的方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param requestString 请求数据
	 * @param method 请求方式，如果requestString不为null，则method将被处理为post
	 * @return
	 */
	public static byte[] getResponseData0(String urlLink, Map<String, String> headers, byte[] requestBytes, String method) throws Exception {
		URL url = new URL(urlLink);
		HttpURLConnection conn = null;
		InputStream is = null;
		try {
			conn = (HttpURLConnection)url.openConnection();
			conn.setRequestMethod(method);

			//设置头
			if(null != headers && headers.size() > 0) {
				for(String key : headers.keySet()) {
					conn.setRequestProperty(key, headers.get(key));
				}
			}

			conn.setDoOutput(true);
			conn.connect();

			//发送数据
			if(null != requestBytes) {
				OutputStream output = conn.getOutputStream();
				output.write(requestBytes, 0, requestBytes.length);
				output.flush();

				FileOperate.close(output);
			}

			//接收数据
			is = conn.getInputStream();
			ByteArrayOutputStream baos = new ByteArrayOutputStream();

			int bufferLen = 4096;
			byte[] data = new byte[bufferLen];
			int readLen;
			while ((readLen = is.read(data, 0, bufferLen)) != -1) {
				baos.write(data, 0, readLen);
			}

			return baos.toByteArray();
		} finally {
			FileOperate.close(is);

			//关闭连接
			if(null != conn) {
				conn.disconnect();
			}
		}
	}

	public static String urlEncode(String value) throws Exception {
		return URLEncoder.encode(value, OConstants.DEFAULT_ENCODING);
	}

	public static void main(String[] args) {
		//for(int i = 0; i < 100; i++) {
			try {
				String url = "http://dl.oppo.com/d/files/skineditor/TCTM/module/OPPO%E6%89%8B%E6%9C%BA%20%E4%B8%BB%E9%A2%98%E5%A5%97%E4%BB%B6.exe";
				byte[] data = HttpDownload.getResponseByGet(url);

				FileOperate.saveContentToFile("g:\\e.exe", data);
			} catch(Exception ex) {
				ex.printStackTrace();
			}
		//}
	}
}