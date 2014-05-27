/**
 * HttpResponseHandler.java
 * com.nearme.base.balance.netty
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

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.concurrent.Executor;

import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.channel.ChannelStateEvent;
import org.jboss.netty.channel.ExceptionEvent;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.channel.SimpleChannelUpstreamHandler;
import org.jboss.netty.handler.codec.http.HttpResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ClassName:HttpResponseHandler <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-3-27  下午2:43:47
 */
public class HttpResponseHandler extends SimpleChannelUpstreamHandler {
	private static Logger logger = LoggerFactory.getLogger(HttpResponseHandler.class);

	private INettyHttpRequest httpRequest;
	private Executor executor;

	public HttpResponseHandler(INettyHttpRequest httpRequest, Executor executor) {
		this.httpRequest = httpRequest;
		this.executor = executor;
	}

	public void messageReceived(final ChannelHandlerContext ctx, MessageEvent e) throws Exception {
		final Object obj = e.getMessage();
		if(obj instanceof HttpResponse){
			if(null == executor) {
				httpRequest.putResponse(ctx.getChannel(), (HttpResponse)obj, true);
			} else {
				executor.execute(new Runnable() {
					@Override
					public void run() {
						httpRequest.putResponse(ctx.getChannel(), (HttpResponse)obj, true);
					}
				});
			}
		} else{
			throw new Exception("仅支持指定的返回类型:HttpResponse:" + ((null == obj) ? "null" : obj.getClass().getSimpleName()));
		}
	}

	public void exceptionCaught(ChannelHandlerContext ctx, ExceptionEvent e)
			throws Exception {
		if(!(e.getCause() instanceof IOException)){
			InetSocketAddress addr = (InetSocketAddress)ctx.getChannel().getRemoteAddress();
			logger.error("系统错误:" + addr.getHostName() + ':' + addr.getPort(), e.getCause());
		}
	}

	public void channelClosed(ChannelHandlerContext ctx, ChannelStateEvent e)
			throws Exception {
		InetSocketAddress addr = (InetSocketAddress)ctx.getChannel().getRemoteAddress();
		if (null != addr) {
			logger.debug("channel关闭: {}:{}", addr.getHostName(), addr.getPort());
		} else {
			logger.debug("channel关闭: " + addr);
		}
	}
}

