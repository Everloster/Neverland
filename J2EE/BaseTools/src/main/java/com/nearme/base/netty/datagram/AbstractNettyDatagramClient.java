/**
 * Copyright (c) 2013 NearMe, All Rights Reserved.
 * FileName:AbstractNettyDatagramClient.java
 * ProjectName:NearmeBaseToolsJ
 * PackageName:com.nearme.base.netty.datagram
 * Create Date:2013-8-22
 * History:
 *   ver	date	  author		desc	
 * ────────────────────────────────────────────────────────
 *   1.0	2013-8-22	  80036381		
 *
 * 
*/

package com.nearme.base.netty.datagram;

import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import org.jboss.netty.bootstrap.ConnectionlessBootstrap;
import org.jboss.netty.channel.ChannelFuture;
import org.jboss.netty.channel.ChannelFutureListener;
import org.jboss.netty.channel.ChannelPipelineFactory;
import org.jboss.netty.channel.socket.DatagramChannel;
import org.jboss.netty.channel.socket.nio.NioDatagramChannelFactory;

import com.nearme.base.concurrent.ExecutorManager;

/**
 * ClassName:AbstractNettyDatagramClient
 * Function: 
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2013-8-22  上午10:49:58
 */
public abstract class AbstractNettyDatagramClient {
	private static final SocketAddress DEFAULT_SOCKET = new InetSocketAddress(0);
	
	private ConnectionlessBootstrap bootstrap;
	private Map<String, Object> options;
	private SocketAddress localAddress;
	private DatagramChannel serverChannel;
	
	private AtomicBoolean initStat = new AtomicBoolean(false);
	
	public AbstractNettyDatagramClient() {
	}
	
	/**
	 * 设置连接选项
	 * @param 
	 * @return
	 */
	public void setOption(String key, Object value) {
		if (null == options) {
			options = new HashMap<String, Object>();
		}
		
		options.put(key, value);
	}
	
	/**
	 * 初始化，如果是接收端，必须主动调用此方法，且调用前需设置好localAddress
	 * @param 
	 * @return
	 */
	public void init() {
		if (null != bootstrap || !initStat.compareAndSet(false, true)) {
			return;
		}

		bootstrap = new ConnectionlessBootstrap(new NioDatagramChannelFactory(
				ExecutorManager.newCachedThreadPool("datagram-worker")));
		bootstrap.setPipelineFactory(getPipelineFactory());
		if (null != options) {
			bootstrap.setOptions(options);
		}
		
		if (null != localAddress) {
			serverChannel = (DatagramChannel)bootstrap.bind(localAddress);
		}
	}
	
	protected abstract ChannelPipelineFactory getPipelineFactory();
	
	/**
	 * 发送数据报文
	 * @param 
	 * @return
	 */
	public void sendDatagram(byte[] message, SocketAddress sockAddr) {
		init();
		
		SocketAddress bindAddr = (null == localAddress) ? DEFAULT_SOCKET : localAddress;
		DatagramChannel datagramChannel = (DatagramChannel)bootstrap.bind(bindAddr);
		datagramChannel.write(message, sockAddr).addListener(new ChannelFutureListener() {
			@Override
			public void operationComplete(ChannelFuture future)
					throws Exception {
				future.getChannel().close();
			}
		});
	}
	
	public void release() {
		if (null != bootstrap) {
			if (null != serverChannel) {
				serverChannel.close();
				serverChannel = null;
			}
			
			bootstrap.releaseExternalResources();
			bootstrap = null;
		}
	}

	/**
	 * 获取localAddress
	 * @return  sth.
	 * @since   Ver 1.0
	 */
	public SocketAddress getLocalAddress() {
		return localAddress;
	}

	/**
	 * 设置localAddress  
	 * @param   localAddress	sth.   
	 * @since   Ver 1.0
	 */
	public void setLocalAddress(SocketAddress localAddress) {
		this.localAddress = localAddress;
	}
}

