/**
 * NettyBusinessServerHandler.java
 * com.nearme.base.netty.server
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-4-26 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.server;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.Executor;

import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelFuture;
import org.jboss.netty.channel.ChannelFutureListener;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.channel.ExceptionEvent;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.channel.SimpleChannelUpstreamHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nearme.base.netty.common.DataHelper;
import com.nearme.base.netty.common.NettyConstants;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.RequestData;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ResponseData;
import com.nearme.base.stat.ExecutorStat;
import com.nearme.base.stat.ExecutorStatPool;

/**
 * ClassName:NettyBusinessServerHandler <br>
 * Function: 业务层数据处理 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-4-26  下午10:16:06
 */
public class NettyBusinessServerHandler extends SimpleChannelUpstreamHandler {

	private static final Logger logger = LoggerFactory.getLogger(NettyBusinessServerHandler.class);

	private IBusinessFactory factory;
	private Executor executor;

	private static final RequestData EMPTY_REQUEST_DATA = RequestData.getDefaultInstance();

	public NettyBusinessServerHandler() {
		this(null, null);
	}

	public NettyBusinessServerHandler(IBusinessFactory factory) {
		this(factory, null);
	}

	/**
	 * 所有请求采用异步处理的方式执行,如果executor为null,则不启用异步处理
	 *
	 * @param executor
	 */
	public NettyBusinessServerHandler(Executor executor) {
		this(null, executor);
	}

	/**
	 * 所有请求采用异步处理的方式执行,如果executor为null,则不启用异步处理
	 *
	 * @param executor
	 */
	public NettyBusinessServerHandler(IBusinessFactory factory, Executor executor) {
		this.executor = executor;

		if (null != factory) {
			this.factory = factory;
		} else {
			this.factory = NettyBusinessFactory.getDefaultBusinessFactory();
		}
	}

	/**
	 * 消息接收并分发（这里也可以考虑做成异步）
	 * @see org.jboss.netty.channel.SimpleChannelUpstreamHandler#messageReceived(org.jboss.netty.channel.ChannelHandlerContext, org.jboss.netty.channel.MessageEvent)
	 */
	@Override
	public void messageReceived(final ChannelHandlerContext ctx, final MessageEvent e)
			throws Exception {
		Object message = e.getMessage();
		if(message instanceof RequestData) {
			//处理单个请求
			handleRequest0(ctx, e, (RequestData)message);
		} else if(message instanceof List) {
			//处理多个请求
			@SuppressWarnings("unchecked")
			final List<RequestData> request = (List<RequestData>)message;
			for(int i = 0, size = request.size(); i < size; i++) {
				handleRequest0(ctx, e, request.get(i));
			}
		} else {
			logger.error("错误的请求", new Exception("仅支持指定的类型:RequestData:"
					+ (null == message ? "null" : message.getClass().getSimpleName())));

			ResponseData response = getEmptyResponse(EMPTY_REQUEST_DATA, NettyConstants.RESPONSE_ILLEAGEL_PROTOCOL, "不合法的数据协议");
			output(ctx, response);
		}
	}

	protected void handleRequest0(final ChannelHandlerContext ctx, final MessageEvent e, final RequestData request) {
		if(null == executor) {
			handleRequest(ctx, e, request);
		} else {
			//采取异步执行的方式
			executor.execute(new Runnable() {
				@Override
				public void run() {
					handleRequest(ctx, e, request);
				}
			});
		}
	}

	protected void handleRequest(ChannelHandlerContext ctx, MessageEvent e, RequestData request) {
		IBusinessHandler businessHandler = factory.getBusinessHandler(request.getRequestType());
		if(null == businessHandler) {
			logger.error("未注册的业务类型:" + request.getRequestType(), new Exception());
			ResponseData response = getEmptyResponse(request, NettyConstants.RESPONSE_UNKNOW_TYPE, "未注册的业务类型");
			output(ctx, response);
			return;
		}

		long start = System.currentTimeMillis();	//处理开始时间
		ResponseData response = null;
		try {
			response = businessHandler.handleRequest(request, e);
			if(null == response && businessHandler.isAsync()) {
				//如果业务处理采用异步方式且无返回，则此处不做处理直接返回,
				//但如果有异常捕获到，则会按异常逻辑处理，不用再直接返回.
				return;
			}
		} catch(Exception ex) {
			//如果执行到这里说明程序问题很大了，必须检查业务处理
			response = getEmptyResponse(request, NettyConstants.RESPONSE_UNKNOW_EXCEPTION, "业务处理出错");
			logger.error("业务处理出错:" + request.getRequestType(), ex);
		}
		
		long total = System.currentTimeMillis() - start;	//处理耗时
		boolean execSucc = (null != response) && response.getCode() >= NettyConstants.RESPONSE_NORMAL;

		//如果超时则不返回
		if(request.getTimeout() > 0 && total >= request.getTimeout()) {
			logger.warn("处理超时,requestType:" + request.getRequestType() + ",id:" + request.getId() + ",ip:" + ctx.getChannel().getRemoteAddress()
					+ ",耗时:" + total + ",超时时长:" + request.getTimeout());
			close(e.getChannel());
			
			execSucc = false;
		} else {
			if(null == response) {
				response = getEmptyResponse(request, NettyConstants.RESPONSE_RESPONSE_NULL, "未获取到返回信息");
			}
			//byte[] responseBytes = response.toByteArray();
			output(ctx, response);
		}
		
		//统计执行时间
		ExecutorStat stat = ExecutorStatPool.getInstance().getExecutorStat(businessHandler.getClass().getName());
		stat.addStat(execSucc, total);
	}

	protected void output(ChannelHandlerContext ctx, ResponseData response) {
		ChannelFuture future = ctx.getChannel().write(response);
		future.addListener(new ChannelFutureListener() {
			@Override
			public void operationComplete(ChannelFuture future) throws Exception {
				//是否主动关闭呢
				close(future.getChannel());
			}
		});
	}

	/**
	 * 返回一个不带数据的返回
	 * @param
	 * @return
	 */
	public ResponseData getEmptyResponse(RequestData request, int responseCode, String responseMsg) {
		return DataHelper.getResponseData(request, request.getRequestType(), responseCode, responseMsg, DataHelper.EMPTY_DATA);
	}

	/**
	 * 异常捕获
	 * @see org.jboss.netty.channel.SimpleChannelUpstreamHandler#exceptionCaught(org.jboss.netty.channel.ChannelHandlerContext, org.jboss.netty.channel.ExceptionEvent)
	 */
	@Override
	public void exceptionCaught(ChannelHandlerContext ctx, ExceptionEvent e)
			throws Exception {
		if(!(e.getCause() instanceof IOException)){
			logger.error("系统错误:" + ctx.getChannel().getRemoteAddress(), e.getCause());

			close(e.getChannel());
		}
	}

	protected void close(Channel ch) {
		//和客户端长连接，不提供关闭功能
//		if(null != ch && ch.isOpen()) {
//			ch.close();
//		}
	}
}

