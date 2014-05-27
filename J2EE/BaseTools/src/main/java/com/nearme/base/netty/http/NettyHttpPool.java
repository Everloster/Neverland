/**
 * NettyHttpPool.java
 * com.nearme.base.netty.http
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-4-7 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.http;

import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

import org.jboss.netty.bootstrap.ClientBootstrap;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelFuture;
import org.jboss.netty.channel.ChannelFutureListener;
import org.jboss.netty.channel.ChannelPipeline;
import org.jboss.netty.channel.ChannelPipelineFactory;
import org.jboss.netty.channel.Channels;
import org.jboss.netty.channel.socket.nio.NioClientSocketChannelFactory;
import org.jboss.netty.handler.codec.http.HttpChunkAggregator;
import org.jboss.netty.handler.codec.http.HttpClientCodec;
import org.jboss.netty.handler.codec.http.HttpContentDecompressor;
import org.jboss.netty.handler.codec.http.HttpHeaders;
import org.jboss.netty.handler.codec.http.HttpHeaders.Names;
import org.jboss.netty.handler.codec.http.HttpHeaders.Values;
import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;

import com.nearme.base.concurrent.ExecutorManager;

/**
 * ClassName:NettyHttpPool <br>
 * Function: Netty http连接池 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-4-7  下午2:02:24
 */
public class NettyHttpPool {

	private static final int DEFAULT_MAX_QUEUE_SIZE = 512;
	private static final String DEFAULT_ACCEPT_ENCODING = HttpHeaders.Values.GZIP + ',' + HttpHeaders.Values.DEFLATE;

	private ClientBootstrap bootstrap;

	private AtomicInteger queueSize;	//队列长度
	private ConcurrentLinkedQueue<ChannelContext> httpQueue; //连接队列
	private int maxQueueSize;			//队列最大允许长度

	private INettyHttpRequest httpRequest;
	private ExecutorService executorService;
	private Map<String, Object> optionMap;

	public NettyHttpPool(ExecutorService executorService, INettyHttpRequest httpRequest, Map<String, Object> optionMap) {
		this.executorService = executorService;
		this.httpRequest = httpRequest;
		this.optionMap = optionMap;

		initial();
	}

	private void initial() {
		maxQueueSize = DEFAULT_MAX_QUEUE_SIZE;
		queueSize = new AtomicInteger(0);
		httpQueue = new ConcurrentLinkedQueue<ChannelContext>();

		bootstrap = createBootstrap();
	}

	public ChannelContext getChannelAndRequest(ChannelContext context, HttpRequest request) {
		//发送请求
		ChannelContext newContext = getChannelContextFromPool(context);
		if(newContext == null) {
			ChannelFuture cf = bootstrap.connect(context.getSocketAddress());
			context.setChannel(cf.getChannel());
			newContext = context;

			cf.addListener(new ConnectChannelFutureListener(context, request));
		} else {
			write(newContext, request);
		}

		return newContext;
	}

	protected ClientBootstrap createBootstrap() {
		ClientBootstrap bootstrap = new ClientBootstrap(new NioClientSocketChannelFactory(
				ExecutorManager.newCachedThreadPool("Client-Pool-Boss-Executor"),
				ExecutorManager.newCachedThreadPool("Client-Pool-Worker-Executor")));

		if (null != optionMap) {
			for (String key : optionMap.keySet()) {
				bootstrap.setOption(key, optionMap.get(key));
			}
		}

		bootstrap.setOption("tcpNoDelay", true);
		bootstrap.setOption("keepAlive", true);

		bootstrap.setPipelineFactory(getPipeline());

		return bootstrap;
	}

	protected ChannelPipelineFactory getPipeline() {
		return new ChannelPipelineFactory() {
			@Override
			public ChannelPipeline getPipeline() throws Exception {
				ChannelPipeline pipeline = Channels.pipeline();
				pipeline.addLast("codec", new HttpClientCodec());
				//支持压缩数据的解压
				pipeline.addLast("inflater", new HttpContentDecompressor());
				//chunk包合并
				pipeline.addLast("aggregator", new HttpChunkAggregator(1048576));

				//http处理handler
				pipeline.addLast("handler", new HttpResponseHandler(httpRequest, executorService));
				return pipeline;
			}
		};
	}

	/**
	 * 释放连接，根据返回的http信息判断是否加入池中
	 * @param
	 * @return
	 */
	public void release(ChannelContext context, HttpResponse response) {
		Channel ch = context.getChannel();
		if(null != ch && ch.isConnected()) {
			if(isKeepAlive(response)) {
				//支持keep-alive则放回连接池
				putChannelContextToPool(context);
			} else {
				//不支持keep-alive则直接关掉
				ch.close();
			}
		}
	}

	public boolean isKeepAlive(HttpResponse response) {
		if (response.isChunked()) {
			return false;
		}

		long contentLength = HttpHeaders.getContentLength(response, -1);
		if (contentLength < 0) {
			return false;
		}

		String connection = response.getHeader(Names.CONNECTION);
		if (null == connection) {
			connection = response.getHeader("Proxy-Connection");
		}

		if (Values.CLOSE.equalsIgnoreCase(connection)) {
            return false;
        }

        if (response.getProtocolVersion().isKeepAliveDefault()) {
            return !Values.CLOSE.equalsIgnoreCase(connection);
        } else {
            return Values.KEEP_ALIVE.equalsIgnoreCase(connection);
        }
	}

	/**
	 * 释放连接，直接关闭
	 * @param
	 * @return
	 */
	public void releaseDirect(ChannelContext context) {
		Channel ch = context.getChannel();
		if(null != ch && ch.isConnected()) {
			ch.close();
		}
	}

	/**
	 * 关闭连接池
	 * @param
	 * @return
	 */
	public void close() {
		ChannelContext cc = null;
		while(null != (cc = httpQueue.poll())) {
			cc.getChannel().close();
		}

		queueSize.set(0);

		bootstrap.releaseExternalResources();
	}

	/**
	 * 返回连接池中的连接，如果不存在则返回null
	 * @param
	 * @return
	 */
	private ChannelContext getChannelContextFromPool(ChannelContext context) {
		ChannelContext cc = httpQueue.poll();
		if(null != cc) {
			queueSize.decrementAndGet();
			if(cc.getChannel().isConnected()) {
				return cc;
			}
		}

		return null;
	}

	/**
	 * 将连接状态的Channel放回池中
	 * @param
	 * @return
	 */
	private void putChannelContextToPool(ChannelContext context) {
		int newSize = queueSize.incrementAndGet();
		if(newSize > maxQueueSize) {
			//大于最大连接数则不保存
			context.getChannel().close();
			queueSize.decrementAndGet();
		} else {
			httpQueue.offer(context);
		}
	}

	private void write(ChannelContext context, HttpRequest request) {
		Channel ch = context.getChannel();

		HttpRequest wrapRequest = wrapRequest(request, context);
		ch.write(wrapRequest).addListener(new WriteChannelFutureListener(context));
	}

	/**
	 * 填充必要的信息到http header中
	 * Function Description here
	 * @param
	 * @return
	 */
	protected HttpRequest wrapRequest(HttpRequest request, ChannelContext context) {
		HttpHeaders.setHost(request, context.getHeaderHost());
		//支持长连接
		HttpHeaders.setKeepAlive(request, true);
		//支持gzip压缩
        request.setHeader(HttpHeaders.Names.ACCEPT_ENCODING, DEFAULT_ACCEPT_ENCODING);

        return request;
	}

	/**
	 * 获取最大的池队列大小
	 * @return  the maxQueueSize
	 * @since   Ver 1.0
	 */
	public int getMaxQueueSize() {
		return maxQueueSize;
	}

	/**
	 * 设置最大的池队列大小
	 * @param   maxQueueSize
	 * @since   Ver 1.0
	 */
	public void setMaxQueueSize(int maxQueueSize) {
		this.maxQueueSize = maxQueueSize;
	}

	final class ConnectChannelFutureListener implements ChannelFutureListener {
		private HttpRequest request;
		private ChannelContext context;

		public ConnectChannelFutureListener(ChannelContext context, HttpRequest request) {
			this.context = context;
			this.request = request;
		}

		@Override
		public void operationComplete(ChannelFuture future) throws Exception {
			Channel ch = future.getChannel();
			context.setChannel(ch);
			if(future.isSuccess()) {
				//发送数据
				write(context, request);
			} else {
				//异常数据发送
				httpRequest.sendError(context, "connect fail.");
			}
		}
	}

	final class WriteChannelFutureListener implements ChannelFutureListener {
		private ChannelContext context;

		public WriteChannelFutureListener(ChannelContext context) {
			this.context = context;
		}

		@Override
		public void operationComplete(ChannelFuture future) throws Exception {
			Channel ch = future.getChannel();
			if(future.isSuccess()) {
				//发送数据成功则不再处理
				return;
			} else {
				//已连上则断开
				if(ch.isConnected()) {
					ch.close();
				}

				//异常数据发送
				httpRequest.sendError(context, "write data fail.");
			}
		}
	}
}

