/**
 * ChannelContext.java
 * com.nearme.base.netty.http
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-4-1 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.http;

import java.net.InetSocketAddress;

import org.jboss.netty.channel.Channel;

/**
 * ClassName:ChannelContext <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-4-1  下午6:16:16
 */
public class ChannelContext {
	private Channel channel;
	private InetSocketAddress socketAddress;
	private String identifier;
	private String hostHeader;
	private String host;

	public ChannelContext() {
	}

	public ChannelContext(Channel channel, String host, int port) {
		this.channel = channel;

		setSocketAddress(host, port);
	}

	public ChannelContext(Channel channel, InetSocketAddress socketAddress) {
		this.channel = channel;

		setSocketAddress(socketAddress);
	}

	/**
	 * 获取channel
	 * @return  the channel
	 * @since   Ver 1.0
	 */
	public Channel getChannel() {
		return channel;
	}
	/**
	 * 设置channel
	 * @param   channel
	 * @since   Ver 1.0
	 */
	public void setChannel(Channel channel) {
		this.channel = channel;
	}

	/**
	 * 获取socketAddress
	 * @return  the socketAddress
	 * @since   Ver 1.0
	 */
	public InetSocketAddress getSocketAddress() {
		return socketAddress;
	}

	/**
	 * 设置socketAddress
	 * @param   socketAddress
	 * @since   Ver 1.0
	 */
	public void setSocketAddress(InetSocketAddress socketAddress) {
		this.host = null;
		setSocketAddressInfo(socketAddress);
	}

	/**
	 * 设置socketAddress
	 * @param   socketAddress
	 * @since   Ver 1.0
	 */
	public void setSocketAddress(String host, int port) {
		this.host = host;
		setSocketAddressInfo(new InetSocketAddress(host, port < 0 ? 80 : port));
	}

	private void setSocketAddressInfo(InetSocketAddress socketAddress) {
		this.socketAddress = socketAddress;

		//获得hostName
		String hostName;
		if (null == host) {
			hostName = socketAddress.getHostName();
		} else {
			hostName = host;
		}

		//获得最终的host header
		String headerHost;
		if(socketAddress.getPort() == 80) {
			headerHost = hostName;
		} else {
			headerHost = hostName + ':' + socketAddress.getPort();
		}

		this.hostHeader = headerHost;
		this.identifier = headerHost;
	}

	/**
	 * 获取
	 * @param
	 * @return
	 */
	public String getHeaderHost() {
		return hostHeader;
	}

	public String getIdentifier() {
		return identifier;
	}
}

