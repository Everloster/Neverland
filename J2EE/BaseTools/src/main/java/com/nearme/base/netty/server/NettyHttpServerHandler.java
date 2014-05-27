/**
 * NettyHttpServerHandler.java
 * com.nearme.base.netty.server
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-2-11 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.server;

import static org.jboss.netty.handler.codec.http.HttpHeaders.Names.CONTENT_TYPE;
import static org.jboss.netty.handler.codec.http.HttpResponseStatus.BAD_REQUEST;
import static org.jboss.netty.handler.codec.http.HttpResponseStatus.INTERNAL_SERVER_ERROR;
import static org.jboss.netty.handler.codec.http.HttpResponseStatus.NOT_FOUND;
import static org.jboss.netty.handler.codec.http.HttpResponseStatus.OK;
import static org.jboss.netty.handler.codec.http.HttpResponseStatus.REQUEST_ENTITY_TOO_LARGE;
import static org.jboss.netty.handler.codec.http.HttpVersion.HTTP_1_1;
import static org.jboss.netty.handler.codec.http.HttpVersion.HTTP_1_0;

import java.io.IOException;
import java.util.concurrent.Executor;

import org.jboss.netty.buffer.ChannelBuffers;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.channel.ExceptionEvent;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.channel.SimpleChannelUpstreamHandler;
import org.jboss.netty.handler.codec.frame.TooLongFrameException;
import org.jboss.netty.handler.codec.http.DefaultHttpResponse;
import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;
import org.jboss.netty.handler.codec.http.HttpResponseStatus;
import org.jboss.netty.util.CharsetUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * ClassName:NettyHttpServerHandler <br>
 * Function: 模拟的http server处理 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-2-11  下午06:26:06
 */
public class NettyHttpServerHandler extends SimpleChannelUpstreamHandler {

	private static final Logger logger = LoggerFactory.getLogger(NettyHttpServerHandler.class);
	private IHttpFactory httpFactory;
	private Executor executor;
	private boolean keepAlive;

	public NettyHttpServerHandler() {
		this(null, null, false);
	}

	/**
	 * 所有请求采用异步处理的方式执行,如果executor为null,则不启用异步处理
	 *
	 * @param executor
	 */
	public NettyHttpServerHandler(Executor executor, boolean keepAlive) {
		this(null, executor, keepAlive);
	}

	/**
	 * 所有请求采用异步处理的方式执行,如果executor为null,则不启用异步处理
	 *
	 * @param executor
	 */
	public NettyHttpServerHandler(IHttpFactory httpFactory, Executor executor, boolean keepAlive) {
		this.executor = executor;
		this.keepAlive = keepAlive;
		if (null == httpFactory) {
			this.httpFactory = DefaultHttpFactory.getDefaultHttpFactory();
		} else {
			this.httpFactory = httpFactory;
		}
	}

	/**
	 * 消息接收并分发（这里也可以考虑做成异步）
	 * @see org.jboss.netty.channel.SimpleChannelUpstreamHandler#messageReceived(org.jboss.netty.channel.ChannelHandlerContext, org.jboss.netty.channel.MessageEvent)
	 */
	@Override
	public void messageReceived(final ChannelHandlerContext ctx, final MessageEvent e)
			throws Exception {
		if(null == executor) {
			handlerRequest(ctx, e);
		} else {
			//采取异步执行的方式
			executor.execute(new Runnable() {
				@Override
				public void run() {
					handlerRequest(ctx, e);
				}
			});
		}
	}

	protected void handlerRequest(ChannelHandlerContext ctx, MessageEvent e) {
		Object msg = e.getMessage();
		if(!(msg instanceof HttpRequest)) {
			sendError(ctx, REQUEST_ENTITY_TOO_LARGE);

			//错误日志输出
			logger.error("接收信息时发生错误，收到的信息类型为:" + (null == msg ? "null" : msg.getClass().getName()));
			return;
		}

		HttpRequest request = (HttpRequest)msg;

		//根据链接转发到不同的处理类中
		String uri = request.getUri();
		IRequestHandler requestHandler = this.httpFactory.getRequestHandler(uri);
		if(null == requestHandler) {
			sendError(ctx, NOT_FOUND);
		} else {
			try {
				//开启keepAlive则使用1.1协议，否则使用1.0协议
				HttpResponse response = new DefaultHttpResponse(keepAlive ? HTTP_1_1 : HTTP_1_0, OK);
				requestHandler.handleRequest(request, response, e);
			} catch(Exception ex) {
				logger.error("请求处理异常:" + uri, ex);
				sendError(ctx, INTERNAL_SERVER_ERROR);
			}
		}
	}

	/**
	 * 异常捕获
	 * @see org.jboss.netty.channel.SimpleChannelUpstreamHandler#exceptionCaught(org.jboss.netty.channel.ChannelHandlerContext, org.jboss.netty.channel.ExceptionEvent)
	 */
	@Override
	public void exceptionCaught(ChannelHandlerContext ctx, ExceptionEvent e)
			throws Exception {
		Channel ch = e.getChannel();
		Throwable cause = e.getCause();
		if(cause instanceof IOException) {
			//可能是超时被关闭或者客户端断开连接
			if(ch.isConnected()) {
				ch.close();
			}
			return;
		}

		//错误日志输出
		logger.error("系统错误:" + ch.getRemoteAddress(), cause);

		//发送错误代码
		if(cause instanceof TooLongFrameException) {
			sendError(ctx, BAD_REQUEST);
		} else {
			sendError(ctx, INTERNAL_SERVER_ERROR);
		}
	}

	private void sendError(ChannelHandlerContext ctx, HttpResponseStatus status) {
		//模拟错误头
		HttpResponse response = new DefaultHttpResponse(HTTP_1_1, status);
		response.setHeader(CONTENT_TYPE, "text/plain; charset=UTF-8");
		response.setContent(ChannelBuffers.wrappedBuffer(("Failure: " + status.toString() + "\r\n").getBytes(CharsetUtil.UTF_8)));

		NettyHttpOutput.output(response, ctx.getChannel());
	}
}

