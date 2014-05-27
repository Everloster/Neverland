/**
 * AbstractNettyBusinessClient.java
 * com.nearme.base.netty.client
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-5-4 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.client;

import java.net.InetSocketAddress;

import org.jboss.netty.bootstrap.Bootstrap;
import org.jboss.netty.channel.ChannelFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ClassName:AbstractNettyBusinessClient <br>
 * Function: 与netty搭建的业务层server端通讯的抽象客户端 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-5-4  上午11:19:49
 */
public abstract class AbstractNettyBusinessClient<T, D> implements INettyClient<T, D> {
	private Logger logger = LoggerFactory.getLogger(this.getClass());

	protected Bootstrap bootstrap;
	protected ChannelFuture cf;
	protected String identifier;

	public AbstractNettyBusinessClient(Bootstrap bootstrap, ChannelFuture cf, String identifier) {
		this.bootstrap = bootstrap;
		this.cf = cf;
		this.identifier = identifier;
	}

	@Override
	public void stop() {
		this.bootstrap.releaseExternalResources();
	}

	/**
	 * 获取服务端标识
	 * @param
	 * @return
	 */
	public String getIdentifier() {
		return this.identifier;
	}

	protected ChannelFuture getChannelFuture() {
		return cf;
	}

	/**
	 * 获取服务端IP
	 * @param
	 * @return
	 */
	public String getServerIP() {
		return ((InetSocketAddress) cf.getChannel().getRemoteAddress()).getHostName();
	}

	/**
	 * 获取服务端端口
	 * @param
	 * @return
	 */
	public int getServerPort() {
		return ((InetSocketAddress) cf.getChannel().getRemoteAddress()).getPort();
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

	/**
	 * 警告日志输出
	 * @param
	 * @return
	 */
	protected void warn(String format, Object[] args) {
		logger.warn(format, args);
	}
}

