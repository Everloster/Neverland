/**
 * INettyHttpRequest.java
 * com.nearme.base.netty.http
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-3-29 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.http;

import java.util.Map;

import org.jboss.netty.channel.Channel;
import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;

/**
 * ClassName:INettyHttpRequest <br>
 * Function: 使用netty实现的http请求 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-3-29  上午9:09:41
 */
public interface INettyHttpRequest {
	/**
	 * 数据返回时的回调方法
	 * @param
	 * @return
	 */
	void putResponse(Channel channel, HttpResponse response, boolean success);

	/**
	 * 产生错误数据时的回调方法
	 * @param
	 * @return
	 */
	void sendError(ChannelContext channelContext, String msg);

	/**
	 * 使用get方式请求http数据
	 * @param url 请求的url
	 * @param headers 请求时的header数据
	 * @param listener 数据返回的回调事件
	 * @return
	 */
	void getResponseByGet(String urlString, Map<String, String> headers, HttpResponseListener listener) throws Exception;

	/**
	 * 使用get方式请求http数据
	 * @param url 请求的url
	 * @param headers 请求时的header数据
	 * @param listener 数据返回的回调事件
	 * @return
	 */
	void getResponseByGet(String urlString, Map<String, String> headers, HttpResponseListener listener, int timeout) throws Exception;

	/**
	 * 使用get方式请求http数据
	 * @param host 请求的host,如store.nearme.com.cn
	 * @param port 请求地址的端口，如80
	 * @param headers 请求时的header数据
	 * @param listener 数据返回的回调事件
	 * @return
	 */
	void getResponseByGet(String host, int port, String uri, Map<String, String> headers,
			HttpResponseListener listener) throws Exception;

	/**
	 * 使用get方式请求http数据
	 * @param host 请求的host,如store.nearme.com.cn
	 * @param port 请求地址的端口，如80
	 * @param headers 请求时的header数据
	 * @param listener 数据返回的回调事件
	 * @param reqTimeout 请求超时时间
	 * @return
	 */
	void getResponseByGet(String host, int port, String uri, Map<String, String> headers,
			HttpResponseListener listener, long reqTimeout) throws Exception;

	/**
	 * 使用host方式请求http数据
	 * @param urlString url路径
	 * @param headers 请求时的header数据
	 * @param requestBytes 请求发送的数据
	 * @param listener 数据返回的回调事件
	 * @return
	 */
	void getResponseByPost(String urlString, Map<String, String> headers, byte[] requestBytes,
			HttpResponseListener listener) throws Exception;

	/**
	 * 使用host方式请求http数据
	 * @param host 请求的host,如store.nearme.com.cn
	 * @param port 请求地址的端口，如80
	 * @param headers 请求时的header数据
	 * @param requestBytes 请求发送的数据
	 * @param listener 数据返回的回调事件
	 * @return
	 */
	void getResponseByPost(String host, int port, String uri, Map<String, String> headers, byte[] requestBytes,
			HttpResponseListener listener) throws Exception;

	/**
	 * 使用host方式请求http数据
	 * @param urlString url路径
	 * @param headers 请求时的header数据
	 * @param requestBytes 请求发送的数据
	 * @param listener 数据返回的回调事件
	 * @return
	 */
	void getResponseByPost(String urlString, Map<String, String> headers, byte[] requestBytes,
			HttpResponseListener listener, long reqTimeout) throws Exception;

	/**
	 * 使用host方式请求http数据
	 * @param host 请求的host,如store.nearme.com.cn
	 * @param port 请求地址的端口，如80
	 * @param headers 请求时的header数据
	 * @param requestBytes 请求发送的数据
	 * @param listener 数据返回的回调事件
	 * @param reqTimeout 请求超时时间
	 * @return
	 */
	void getResponseByPost(String host, int port, String uri, Map<String, String> headers, byte[] requestBytes,
			HttpResponseListener listener, long reqTimeout) throws Exception;

	/**
	 *
	 * @param
	 * @return
	 */
	void sendRequest(final String host, int port, final HttpRequest request,
			HttpResponseListener listener, long reqTimeout);
}