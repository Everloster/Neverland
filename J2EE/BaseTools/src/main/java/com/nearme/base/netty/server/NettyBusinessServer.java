/**
 * NettyBusinessServer.java
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

import com.nearme.base.netty.codec.ProtobufInOneDecoder;
import com.nearme.base.netty.codec.ProtobufInOneEncoder;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.RequestData;


/**
 * ClassName:NettyBusinessServer <br>
 * Function: 由netty搭建的业务层server端 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-4-26  下午09:42:08
 */
public class NettyBusinessServer extends AbstractNettyServer {
	@Override
	public ChannelPipelineFactory getChannelPipelineFactory(final Executor handlerExecutor) {
		return new ChannelPipelineFactory() {
			@Override
			public ChannelPipeline getPipeline() throws Exception {
				ChannelPipeline pipeline = Channels.pipeline();
				pipeline.addLast("pbInOneDecoder", new ProtobufInOneDecoder(RequestData.getDefaultInstance()));
				pipeline.addLast("pbInOneEncoder", new ProtobufInOneEncoder());
				
				pipeline.addLast("handler", getHttpServerHandler(handlerExecutor));
				return pipeline;
			}
		};
	}
	
	/**
	 * 获取业务处理类
	 * @param 
	 * @return
	 */
	public SimpleChannelUpstreamHandler getHttpServerHandler(final Executor handlerExecutor) {
		return new NettyBusinessServerHandler(handlerExecutor);
	}
}

