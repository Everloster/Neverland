/**
 * HttpBalanceRequest.java
 * com.nearme.base.balance
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-8-15 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance;

import java.io.File;
import java.io.InputStream;
import java.util.Map;

import com.nearme.base.balance.model.ServerInfo;
import com.nearme.base.http.HttpConnectUtil;
import com.oppo.base.common.OConstants;

/**
 * ClassName:HttpBalanceRequest <br>
 * Function: 按照负载配置进行分流的http请求 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-8-15  下午07:58:02
 */
public class HttpBalanceRequest extends AbstractBalanceRequest {

	/**
	 * 文件方式的加载，此处默认为xml文件
	 *
	 * @param configFile
	 */
	public HttpBalanceRequest(File configFile) {
		init(configFile);
	}

	/**
	 * InputStream方式的加载，此处默认为xml的InputStream
	 *
	 * @param inputStream
	 */
	public HttpBalanceRequest(InputStream inputStream) {
		init(inputStream);
	}

	/**
	 * 使用get方式请求http数据
	 * @param url 请求的url
	 * @return
	 */
	public byte[] getResponseByGet(String groupId, String uri, Map<String, String> headers) throws Exception {
		ServerInfo server = this.getConnectServer(groupId);
		if(null == server) {
			return null;
		}

		String url = getRequestUrl(server.getServerUrl(), uri);
		try {
			byte[] data = HttpConnectUtil.getResponseByGet(url, headers);

			stat(groupId, server, null != data);
			return data;
		} catch(Exception ex) {
			stat(groupId, server, false);
			throw new Exception("url:" + url, ex);
		}
	}

	/**
	 * 使用post方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param requestString 请求数据
	 * @return
	 */
	public byte[] getResponseByPost(String groupId, String urlSuffix, Map<String, String> headers, String requestString) throws Exception {
		return getResponseByPost(groupId, urlSuffix, headers, requestString.getBytes(OConstants.DEFAULT_ENCODING));
	}

	/**
	 * 使用指定的方式请求http数据
	 * @param urlLink 请求链接
	 * @param headers 请求头
	 * @param requestString 请求数据
	 * @return
	 */
	public byte[] getResponseByPost(String groupId, String urlSuffix, Map<String, String> headers, byte[] requestBytes) throws Exception {
		ServerInfo server = this.getConnectServer(groupId);
		if(null == server) {
			return null;
		}

		String url = getRequestUrl(server.getServerUrl(), urlSuffix);

		try {
			byte[] data = HttpConnectUtil.getResponseByPost(url, headers, requestBytes);

			stat(groupId, server, null != data);
			return data;
		} catch(Exception ex) {
			stat(groupId, server, false);

			throw new Exception("url:" + url, ex);
		}
	}

	/**
	 * 获取完整的请求连接
	 * @param
	 * @return
	 */
	protected String getRequestUrl(String serverUrl, String urlSuffix) {
		return (null == urlSuffix) ? serverUrl : (serverUrl + urlSuffix);
	}
}