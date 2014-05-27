/**
 * NettyHttpServer.java
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

import java.util.concurrent.Executor;

import org.jboss.netty.channel.ChannelPipeline;
import org.jboss.netty.channel.ChannelPipelineFactory;
import org.jboss.netty.channel.Channels;
import org.jboss.netty.channel.SimpleChannelUpstreamHandler;
import org.jboss.netty.handler.codec.http.HttpChunkAggregator;
import org.jboss.netty.handler.codec.http.HttpRequestDecoder;
import org.jboss.netty.handler.codec.http.HttpResponseEncoder;
import org.jboss.netty.handler.timeout.IdleStateAwareChannelHandler;
import org.jboss.netty.handler.timeout.IdleStateHandler;

import com.nearme.base.netty.timeout.HttpHeartbeatHandler;


/**
 * ClassName:NettyHttpServer <br>
 * Function: 由netty搭建的接入层server端 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-4-26  下午09:48:39
 */
public class NettyHttpServer extends AbstractNettyServer {
	private IdleStateHandler idleStateHandler;
	private IdleStateAwareChannelHandler idleStateAwareHandler;

	private int maxAggregatorLength;

	public NettyHttpServer() {
		this(null);
	}

	public NettyHttpServer(IdleStateHandler idelStatHandler) {
		this(idelStatHandler, (null == idelStatHandler) ? null : new HttpHeartbeatHandler());
	}

	public NettyHttpServer(IdleStateHandler idelStatHandler, IdleStateAwareChannelHandler idleStateAwareHandler) {
		super();

		this.idleStateHandler = idelStatHandler;
		this.idleStateAwareHandler = idleStateAwareHandler;

		if(null != idelStatHandler) {
			this.setKeepAlive(true);
		}
	}

	@Override
	public ChannelPipelineFactory getChannelPipelineFactory(final Executor handlerExecutor) {
		return new ChannelPipelineFactory(){
			@Override
			public ChannelPipeline getPipeline() throws Exception {
				return getChannelPipeline(handlerExecutor);
			}
		};
	}

	/**
	 * 获取pipeline
	 * @param
	 * @return
	 */
	public ChannelPipeline getChannelPipeline(Executor handlerExecutor) {
		ChannelPipeline pipeline = Channels.pipeline();

		if (null != idleStateHandler) {
			//keep alive
			pipeline.addLast("idelStat", idleStateHandler);
			pipeline.addLast("heartbeat", idleStateAwareHandler);
		}

		pipeline.addLast("decoder", new HttpRequestDecoder());

		//将块信息组合为一个完整的HttpMessage
		if(this.maxAggregatorLength > 0) {
			pipeline.addLast("aggregator", new HttpChunkAggregator(this.maxAggregatorLength));
		}

		pipeline.addLast("encoder", new HttpResponseEncoder());

		//超时管理
		//pipeline.addLast("timeout", new IdleStateHandler(timer, 10, 10, 0));//此两项为添加心跳机制 10秒查看一次在线的客户端channel是否空闲
		//pipeline.addLast("hearbeat", new Heartbeat());//此类 实现了IdleStateAwareChannelHandler接口

		//http处理handler
		pipeline.addLast("handler", getHttpServerHandler(handlerExecutor));
		return pipeline;
	}

	/**
	 * 获取http处理类
	 * @param
	 * @return
	 */
	public SimpleChannelUpstreamHandler getHttpServerHandler(final Executor handlerExecutor) {
		return new NettyHttpServerHandler(handlerExecutor, isKeepAlive());
	}

	/**
	 * 获取最大的聚合HttpMessage信息大小,当该值<=0时,不启用聚合功能
	 * @return  the maxAggregatorLength
	 * @since   Ver 1.0
	 */
	public int getMaxAggregatorLength() {
		return maxAggregatorLength;
	}

	/**
	 * 设置最大的聚合HttpMessage信息大小,当该值<=0时,不启用聚合功能
	 * @param   maxAggregatorLength
	 * @since   Ver 1.0
	 */
	public void setMaxAggregatorLength(int maxAggregatorLength) {
		this.maxAggregatorLength = maxAggregatorLength;
	}
}

