/**
 * ClientAccessHttpServer.java
 * com.nearme.market.access.main
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-9-25 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.gamecenter.ddz.oauth.server;

import java.util.concurrent.Executor;

import org.jboss.netty.channel.ChannelPipeline;
import org.jboss.netty.channel.Channels;
import org.jboss.netty.handler.codec.http.HttpChunkAggregator;
import org.jboss.netty.handler.codec.http.HttpRequestDecoder;
import org.jboss.netty.handler.codec.http.HttpResponseEncoder;

/**
 * ClassName:ClientAccessHttpServer <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-9-25  下午04:20:08
 */
public class ClientAccessHttpServer extends NettyHttpServer {
	/**
	 * @since Ver 1.1
	 */
//	private static final long serialVersionUID = 1L;
	
	/**
	 * 获取pipeline
	 * @param 
	 * @return
	 */
	@Override
	public ChannelPipeline getChannelPipeline(Executor handlerExecutor) {
		ChannelPipeline pipeline = Channels.pipeline();
		pipeline.addLast("decoder", new HttpRequestDecoder());
		//最大支持128k的数据
		pipeline.addLast("aggregator", new HttpChunkAggregator(1024 * 128));
		pipeline.addLast("encoder", new HttpResponseEncoder());
		
		//超时管理
		//pipeline.addLast("timeout", new IdleStateHandler(timer, 10, 10, 0));//此两项为添加心跳机制 10秒查看一次在线的客户端channel是否空闲
		//pipeline.addLast("hearbeat", new Heartbeat());//此类 实现了IdleStateAwareChannelHandler接口
		
		//http处理handler
		pipeline.addLast("handler", getHttpServerHandler(handlerExecutor));
		return pipeline;
	}
}

