/**
 * DefaultNettyHttpRequest.java
 * com.nearme.base.netty.http
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-3-27 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.http;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

import org.jboss.netty.buffer.ChannelBuffers;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.handler.codec.http.DefaultHttpRequest;
import org.jboss.netty.handler.codec.http.DefaultHttpResponse;
import org.jboss.netty.handler.codec.http.HttpHeaders;
import org.jboss.netty.handler.codec.http.HttpMethod;
import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;
import org.jboss.netty.handler.codec.http.HttpResponseStatus;
import org.jboss.netty.handler.codec.http.HttpVersion;
import org.jboss.netty.util.Timeout;
import org.jboss.netty.util.Timer;
import org.jboss.netty.util.TimerTask;

import com.nearme.base.netty.client.NettyHttpContext;
import com.nearme.base.stat.ExecutorStatPool;

/**
 * ClassName:DefaultNettyHttpRequest <br>
 * Function: netty实现的http的请求 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-3-27  下午7:24:45
 */
public class DefaultNettyHttpRequest implements INettyHttpRequest {
	/**
	 * 默认超时时间 10s
	 */
	public static final int DEFAULT_TIMEOUT = 10 * 1000;

	private ConcurrentMap<Channel, RequestContext> channelMap;
	private NettyHttpManager httpManager;
//	private Executor executor;
	private Timer timer;
	private int timeout = DEFAULT_TIMEOUT;

	/**
	 *
	 */
	public DefaultNettyHttpRequest(ExecutorService executorService, Timer timer) {
		init();
		this.httpManager = new NettyHttpManager(this, executorService);
		this.timer = timer;
	}

	protected void init() {
		channelMap = new ConcurrentHashMap<Channel, RequestContext>();
	}

	/**
	 * 使用get方式请求http数据
	 * @param url 请求的url
	 * @param headers 请求时的header数据
	 * @param listener 数据返回的回调事件
	 * @return
	 * @throws MalformedURLException
	 */
	public void getResponseByGet(String urlString, Map<String, String> headers, HttpResponseListener listener) throws MalformedURLException {
		getResponseByGet(urlString, headers, listener, timeout);
	}

	/**
	 * 使用get方式请求http数据
	 * @param url 请求的url
	 * @param headers 请求时的header数据
	 * @param listener 数据返回的回调事件
	 * @return
	 * @throws MalformedURLException
	 */
	public void getResponseByGet(String urlString, Map<String, String> headers, HttpResponseListener listener, int timeout) throws MalformedURLException {
		URL url = new URL(urlString);

		getResponseByGet(url.getHost(), url.getPort(), getUri(url), headers, listener, timeout);
	}

	/**
	 * 使用get方式请求http数据
	 * @param host 请求的host,如store.nearme.com.cn
	 * @param port 请求地址的端口，如80
	 * @param headers 请求时的header数据
	 * @param listener 数据返回的回调事件
	 * @return
	 */
	public void getResponseByGet(String host, int port, String uri, Map<String, String> headers,
			HttpResponseListener listener) {
		getResponseByGet(host, port, uri, headers, listener, timeout);
	}

	/**
	 * 使用get方式请求http数据
	 * @param host 请求的host,如store.nearme.com.cn
	 * @param port 请求地址的端口，如80
	 * @param headers 请求时的header数据
	 * @param listener 数据返回的回调事件
	 * @param reqTimeout 请求超时时间
	 * @return
	 */
	public void getResponseByGet(String host, int port, String uri, Map<String, String> headers,
			HttpResponseListener listener, long reqTimeout) {
		final HttpRequest request = new DefaultHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, uri);
		if(null != headers) {
			for(String key : headers.keySet()) {
				request.addHeader(key, headers.get(key));
			}
		}

		sendRequest(host, port, request, listener, reqTimeout);
	}

	/**
	 * 使用host方式请求http数据
	 * @param urlString url路径
	 * @param headers 请求时的header数据
	 * @param requestBytes 请求发送的数据
	 * @param listener 数据返回的回调事件
	 * @return
	 * @throws MalformedURLException
	 */
	public void getResponseByPost(String urlString, Map<String, String> headers, byte[] requestBytes,
			HttpResponseListener listener) throws MalformedURLException {
		URL url = new URL(urlString);
		getResponseByPost(url.getHost(), url.getPort(), getUri(url), headers, requestBytes, listener);
	}

	/**
	 * 使用host方式请求http数据
	 * @param host 请求的host,如store.nearme.com.cn
	 * @param port 请求地址的端口，如80
	 * @param headers 请求时的header数据
	 * @param requestBytes 请求发送的数据
	 * @param listener 数据返回的回调事件
	 * @return
	 * @throws MalformedURLException
	 */
	public void getResponseByPost(String host, int port, String uri, Map<String, String> headers, byte[] requestBytes,
			HttpResponseListener listener) throws MalformedURLException {
		this.getResponseByPost(host, port, uri, headers, requestBytes, listener, timeout);
	}

	/**
	 * 使用host方式请求http数据
	 * @param urlString url路径
	 * @param headers 请求时的header数据
	 * @param requestBytes 请求发送的数据
	 * @param listener 数据返回的回调事件
	 * @return
	 * @throws MalformedURLException
	 */
	public void getResponseByPost(String urlString, Map<String, String> headers, byte[] requestBytes,
			HttpResponseListener listener, long reqTimeout) throws MalformedURLException {
		URL url = new URL(urlString);
		getResponseByPost(url.getHost(), url.getPort(), getUri(url), headers, requestBytes, listener, reqTimeout);
	}

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
	public void getResponseByPost(String host, int port, String uri, Map<String, String> headers, byte[] requestBytes,
			HttpResponseListener listener, long reqTimeout) {
		HttpRequest request = new DefaultHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, uri);
		if(null != headers) {
			for(String key : headers.keySet()) {
				request.addHeader(key, headers.get(key));
			}
		}

		if (null != requestBytes) {
			//设置长度
			HttpHeaders.setContentLength(request, requestBytes.length);
			request.setContent(ChannelBuffers.wrappedBuffer(requestBytes));
		}

		sendRequest(host, port, request, listener, reqTimeout);
	}

	public void sendRequest(final String host, int port, final HttpRequest request,
			HttpResponseListener listener, long reqTimeout) {
		//请求数据上下文
		NettyHttpContext<HttpResponseListener> httpContext = new NettyHttpContext<HttpResponseListener>();
		httpContext.setStartTime(System.currentTimeMillis());
		httpContext.addServerResponseListener(listener);
		httpContext.setRequest(request);

		ChannelContext initChannelContext = new ChannelContext();
		initChannelContext.setSocketAddress(host, port);
		ChannelContext acquiredContext = httpManager.getChannelAndRequest(initChannelContext, request);

		RequestContext requestContext = new RequestContext(httpContext, acquiredContext);
		//超时检查任务
		Timeout timeout = timer.newTimeout(new RequestTimeoutTask(requestContext, reqTimeout),
				reqTimeout, TimeUnit.MILLISECONDS);
		httpContext.setTimeout(timeout);

		channelMap.put(acquiredContext.getChannel(), requestContext);
	}

	@Override
	public void putResponse(Channel channel, HttpResponse response, boolean success) {
		RequestContext requestContext = channelMap.remove(channel);

		if(null != requestContext) {
			NettyHttpContext<HttpResponseListener> context = requestContext.getHttpContext();
			context.setRequestComplete(true);
			// 取消定时器
			Timeout timeout = context.getTimeout();
			if (null != timeout && !timeout.isCancelled()) { 
				timeout.cancel();
			}
			
			ChannelContext channelContext = requestContext.getChannelContext();
			//执行时间统计
			boolean isSucc = (HttpResponseStatus.OK.equals(response.getStatus()));
			long executeTime = System.currentTimeMillis() - context.getStartTime();
			ExecutorStatPool.getInstance().getExecutorStat(
					channelContext.getIdentifier()).addStat(isSucc, executeTime);

			//执行回调事件
			List<HttpResponseListener> listeners = context.getListeners();
			if(null != listeners) {
				for(int i = 0, size = listeners.size(); i < size; i++) {
					HttpResponseListener listener = listeners.get(i);
					if(isSucc) {
						listener.onSuccess(response);
					} else {
						listener.onError(response);
					}
				}
			}

			//释放连接
			if(success) {
				httpManager.release(channelContext, response);
			} else {
				//如果执行出错则移除
				httpManager.releaseDirect(channelContext);
			}
		} else {
			//无对应关系则忽略
			//System.out.println(response);
		}
	}

	public void sendError(ChannelContext context, String msg) {
		HttpResponse response = new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.INTERNAL_SERVER_ERROR);
		response.setContent(ChannelBuffers.wrappedBuffer(msg.getBytes()));

		//发送错误信息
		putResponse(context.getChannel(), response, false);
	}

	public String getUri(URL url) {
		StringBuilder sb = new StringBuilder();
		if(null != url.getPath()) {
			sb.append(url.getPath());
		}
		if(null != url.getQuery()) {
			sb.append('?').append(url.getQuery());
		}

		if(sb.length() == 0) {
			return "/";
		} else {
			return sb.toString();
		}
	}

	public void close() {
		channelMap.clear();
		timer.stop();
		httpManager.close();
	}

	/**
	 * 获取timeout
	 * @return  the timeout
	 * @since   Ver 1.0
	 */
	public int getTimeout() {
		return timeout;
	}

	/**
	 * 设置timeout
	 * @param   timeout
	 * @since   Ver 1.0
	 */
	public void setTimeout(int timeout) {
		this.timeout = timeout;
	}

	/**
	 * 请求超时任务处理
	 * @author   80036381
	 * @version  NettyBusinessAsyncClient
	 * @since    Ver 1.1
	 * @Date	 2013-3-15  上午9:51:43
	 * @see
	 */
	private final class RequestTimeoutTask implements TimerTask {
		private RequestContext requestContext;
		private long reqTimeout;

		public RequestTimeoutTask(RequestContext requestContext, long reqTimeout) {
			this.requestContext = requestContext;
			this.reqTimeout = reqTimeout;
		}

		@Override
		public void run(Timeout timeout) throws Exception {
			NettyHttpContext<HttpResponseListener> httpContext = requestContext.getHttpContext();
			if (httpContext.isRequestComplete() || timeout.isCancelled()) {
                return;
            }

			//执行时间
			long executeTime = System.currentTimeMillis() - httpContext.getStartTime();

			if(reqTimeout <= executeTime) {
				//超时移除
				HttpResponse response = new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.REQUEST_TIMEOUT);
				putResponse(requestContext.getChannelContext().getChannel(), response, false);
			} else {
				//还未超时则重新处理
				timer.newTimeout(this, reqTimeout - executeTime, TimeUnit.MILLISECONDS);
			}
		}
	}

	private final class RequestContext {
		private NettyHttpContext<HttpResponseListener> httpContext;
		private ChannelContext channelContext;
		public RequestContext(NettyHttpContext<HttpResponseListener> httpContext, ChannelContext channelContext) {
			this.httpContext = httpContext;
			this.channelContext = channelContext;
		}
		/**
		 * 获取httpContext
		 * @return  the httpContext
		 * @since   Ver 1.0
		 */
		public NettyHttpContext<HttpResponseListener> getHttpContext() {
			return httpContext;
		}
		/**
		 * 获取channelContext
		 * @return  the channelContext
		 * @since   Ver 1.0
		 */
		public ChannelContext getChannelContext() {
			return channelContext;
		}
	}
}

