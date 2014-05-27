/**
 * AbstractNettyBusinessClientHandler.java
 * com.nearme.base.netty.client
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-12-12 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.client;

import java.io.IOException;

import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.channel.ChannelStateEvent;
import org.jboss.netty.channel.ExceptionEvent;
import org.jboss.netty.channel.SimpleChannelUpstreamHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ClassName:AbstractNettyBusinessClientHandler <br>
 * Function: netty客户端处理 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-12-12  下午02:21:51
 */
public class AbstractNettyBusinessClientHandler<T, D> extends SimpleChannelUpstreamHandler {
	
	private Logger logger = LoggerFactory.getLogger(this.getClass());
	
	private AbstractNettyBusinessClientFactory<T, D> factory;
	private INettyClient<T, D> client;
	private String identifier;
	
	public AbstractNettyBusinessClientHandler(AbstractNettyBusinessClientFactory<T, D> factory,String identifier){
		this.factory = factory;
		this.identifier = identifier;
	}
	
	public void setClient(INettyClient<T, D> client){
		this.client = client;
	}
	
	public INettyClient<T, D> getClient() {
		return this.client;
	}
	
	public void exceptionCaught(ChannelHandlerContext ctx, ExceptionEvent e)
			throws Exception {
		if(!(e.getCause() instanceof IOException)){
			error("系统错误", e.getCause());
		}
	}
	
	public void channelClosed(ChannelHandlerContext ctx, ChannelStateEvent e)
			throws Exception {
		factory.removeClient(identifier, client, "channel is closed.");
	}
	
	/**
	 * 错误日志输出
	 * @param 
	 * @return
	 */
	protected void error(String msg, Throwable t) {
		logger.error(msg, t);
	}
	
	/**
	 * 警告日志输出
	 * @param 
	 * @return
	 */
	protected void warn(String msg) {
		logger.warn(msg);
	}
}

