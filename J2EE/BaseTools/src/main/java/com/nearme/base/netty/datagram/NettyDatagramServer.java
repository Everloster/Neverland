/**
 * Copyright (c) 2013 NearMe, All Rights Reserved.
 * FileName:NettyDatagramServer.java
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

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.channel.ChannelPipeline;
import org.jboss.netty.channel.ChannelPipelineFactory;
import org.jboss.netty.channel.Channels;
import org.jboss.netty.channel.ExceptionEvent;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.channel.SimpleChannelUpstreamHandler;

import com.nearme.base.concurrent.ExecutorManager;


/**
 * ClassName:NettyDatagramServer
 * Function: TODO ADD FUNCTION
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2013-8-22  下午1:15:09
 */
public class NettyDatagramServer extends AbstractNettyDatagramClient {

	private ExecutorService es;
	private IDatagramService datagramService;
	
	public NettyDatagramServer() {
		this(ExecutorManager.newCachedThreadPool("datagrame-server"), null);
	}
	
	public NettyDatagramServer(ExecutorService es) {
		this(es, null);
	}
	
	public NettyDatagramServer(ExecutorService es, IDatagramService datagramService) {
		super();
		
		this.es = es;
		this.datagramService = datagramService;
	}
	
	@Override
	protected ChannelPipelineFactory getPipelineFactory() {
		return new ChannelPipelineFactory() {
            public ChannelPipeline getPipeline() throws Exception {
                return Channels.pipeline(
                        new BytesEncoder(),
                        new BytesDecoder(),
                        new ServerHandler()
                        );
            }
        };
	}
	
	/**
	 * 获取datagramService
	 * @return  sth.
	 * @since   Ver 1.0
	 */
	public IDatagramService getDatagramService() {
		return datagramService;
	}

	/**
	 * 设置datagramService  
	 * @param   datagramService	sth.   
	 * @since   Ver 1.0
	 */
	public void setDatagramService(IDatagramService datagramService) {
		this.datagramService = datagramService;
	}

	private class ServerHandler extends SimpleChannelUpstreamHandler {
		@Override
		public void messageReceived(ChannelHandlerContext ctx, MessageEvent e)
				throws Exception {
			final byte[] msg = (byte[]) e.getMessage();
			final SocketAddress remoteAddr = e.getRemoteAddress();
			if (null == es) {
				handleData(msg, remoteAddr);
			} else {
				es.submit(new Runnable() {
					@Override
					public void run() {
						handleData(msg, remoteAddr);
					}
				});
			}
		}
		
		private void handleData(byte[] msg, SocketAddress socketAddress) {
			if (null != datagramService) {
				datagramService.handleMessage(msg, socketAddress);
			}
		}

		@Override
		public void exceptionCaught(ChannelHandlerContext ctx, ExceptionEvent e)
				throws Exception {
			Throwable t = e.getCause();
			if (!(t instanceof IOException)) {
				if (null != datagramService) {
					datagramService.handleError(t);
				}
			}
		}
	}
	
	public static void main(String[] args) throws Exception {
		final NettyDatagramServer s = new NettyDatagramServer();
		s.setLocalAddress(new InetSocketAddress(9090));
		final AtomicInteger counter = new AtomicInteger();
		s.setDatagramService(new IDatagramService() {
			@Override
			public void handleMessage(byte[] msg, SocketAddress socketAddress) {
				counter.addAndGet(1);
				System.out.println(new String(msg));
			}

			@Override
			public void handleError(Throwable t) {
				t.printStackTrace();
			}
		});
		s.init();
		
		Thread t = new Thread(new Runnable() {
			@Override
			public void run() {
				while(true) {
					try {
						Thread.sleep(2000);
					} catch (InterruptedException e) {
					}
					
					System.out.println(counter.get());
				}
			}
		});
		t.start();
	}
}
