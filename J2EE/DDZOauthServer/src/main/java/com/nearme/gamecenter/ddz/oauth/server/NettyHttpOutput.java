/**
 * NettyHttpOutput.java
 * com.nearme.base.netty.server
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-4-11 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.gamecenter.ddz.oauth.server;

import org.jboss.netty.buffer.ChannelBuffers;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelFuture;
import org.jboss.netty.channel.ChannelFutureListener;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.handler.codec.http.HttpHeaders;
import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;

/**
 * ClassName:NettyHttpOutput <br>
 * Function: http数据输出 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-4-11  下午8:00:56
 */
public final class NettyHttpOutput {
	/**
	 * 输出指定的byte数组到response中并关闭当前连接
	 * @param data 输出的字节
	 * @param response
	 * @param e
	 * @return
	 */
	public static void output(byte[] data, HttpResponse response, MessageEvent e) {
		//设置输出的数据
		response.setContent(ChannelBuffers.wrappedBuffer(data));

		output(response, null, e.getChannel());
	}

	/**
	 * 输出指定的byte数组到response中并关闭当前连接
	 * @param data 输出的字节
	 * @param response
	 * @param e
	 * @return
	 */
	public static void output(byte[] data, HttpResponse response, Channel ch) {
		//设置输出的数据
		response.setContent(ChannelBuffers.wrappedBuffer(data));

		output(response, null, ch);
	}

	/**
	 * 输出response中的内容但未关闭当前连接
	 * @param
	 * @return
	 */
	public static void output(HttpResponse response, MessageEvent e) {
		output(response, null, e.getChannel());
	}

	/**
	 * 输出response中的内容但未关闭当前连接
	 * @param
	 * @return
	 */
	public static void output(HttpResponse response, Channel ch) {
		output(response, null, ch);
	}

	/**
	 * 输出response中的内容但未关闭当前连接
	 * @param
	 * @return
	 */
	public static void output(HttpResponse response, HttpRequest request, Channel ch) {
		response.setHeader("Content-Length", response.getContent().readableBytes());

		if(null != ch) {
			if(ch.isConnected()) {
				//写入keep-alive头
				boolean keepAlive = HttpHeaders.isKeepAlive(response);
				if (keepAlive && null != request) {
					keepAlive = HttpHeaders.isKeepAlive(request);
				}
//				if(keepAlive) {
//					//显式的设置keep-alive
//					HttpHeaders.setKeepAlive(response, keepAlive);
//				}

				//发送数据
//				ch.write(response).addListener(CHANNEL_CLOSE);
				ChannelFuture cf = ch.write(response);
				if(!keepAlive) {
					//如果不支持keep alive则直接关闭
					cf.addListener(CHANNEL_CLOSE);
				}
			}
		}
	}

	//关闭连接时间
	public static ChannelFutureListener CHANNEL_CLOSE = new ChannelFutureListener() {
		public void operationComplete(ChannelFuture future) {
			Channel c = future.getChannel();
	        if(null != c && c.isConnected()) {
	        	c.close();
	        }
	    }
	};
}

