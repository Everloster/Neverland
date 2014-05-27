/**
 * Copyright (c) 2013 NearMe, All Rights Reserved.
 * FileName:NettyDatagramClient.java
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.jboss.netty.channel.ChannelPipeline;
import org.jboss.netty.channel.ChannelPipelineFactory;
import org.jboss.netty.channel.Channels;

import com.nearme.base.concurrent.ExecutorManager;

/**
 * ClassName:NettyDatagramClient
 * Function: 数据包发送端（不接收数据）
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2013-8-22  上午11:56:52
 */
public class NettyDatagramClient extends AbstractNettyDatagramClient {

	@Override
	protected ChannelPipelineFactory getPipelineFactory() {
		return new ChannelPipelineFactory() {
            public ChannelPipeline getPipeline() throws Exception {
                return Channels.pipeline(
                        new BytesEncoder(),
                        new BytesDecoder());
            }
        };
	}
	
	public static void main(String[] args) throws Exception {
		final NettyDatagramClient c = new NettyDatagramClient();
		c.init();
		ExecutorService es = ExecutorManager.newCachedThreadPool();
		final AtomicInteger a = new  AtomicInteger();
		for (int i = 0; i < 100; i++) {
			final String key = "信息-" + String.valueOf(i) + "_";
			es.execute(new Runnable() {
				public void run() {
					for (int i = 0; i < 100; i++) {
						String k = key + i;
						c.sendDatagram(k.getBytes(), new InetSocketAddress("127.0.0.1", 9090));
						try {
							Thread.sleep(1);
						} catch (InterruptedException e) {
						}
						a.addAndGet(1);
					}
				}
			});
		}
		
		es.awaitTermination(5, TimeUnit.SECONDS);
		System.out.println("ready to close...");
		c.release();
		ExecutorManager.shutdownAll();
		System.out.println("end,msg count:" + a.get());
	}
}

