/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年1月21日
 */
package com.nearme.gamecenter.ddz.redis;

import org.jboss.netty.buffer.ChannelBuffers;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.handler.codec.http.HttpResponse;
import org.jboss.netty.handler.codec.http.HttpResponseStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nearme.gamecenter.ddz.oauth.handler.CoreProxyHandler;
import com.nearme.gamecenter.ddz.oauth.server.NettyHttpOutput;
import com.nearme.gamecenter.ddz.oauth.util.JResultUtil;

/**
 * 
 * @Author	LaiLong
 * @Since	2014年1月21日
 */
public abstract class RedisCallback {
	
	private static final Logger logger = LoggerFactory
			.getLogger(RedisCallback.class);
	
	private HttpResponse response;
	private MessageEvent event;
	private long startTime;
	
	public void fire(Object something) {
		try {
			String result = (String) something;
			if (result.equals(RedisServer.TIME_OUT_STRING)) {
				logger.error("fire message timeout error, message id:" + getIdentify());
				response.setContent(ChannelBuffers
						.wrappedBuffer(JResultUtil.getFailureJsonString(CoreProxyHandler.RESULT_TIMEOUT)
								.getBytes()));
				response.setStatus(HttpResponseStatus.OK);
			} else {
				logger.info("fire message success, message id:" + getIdentify());
				response.setContent(ChannelBuffers
						.wrappedBuffer(JResultUtil.getOkJsonString(result)
								.getBytes()));
				response.setStatus(HttpResponseStatus.OK);
			}
		} catch (Exception e) {
			logger.error("fire message error, e:" + e.getMessage() + " result:" + something);
			response.setContent(ChannelBuffers
					.wrappedBuffer(JResultUtil.getFailureJsonString(CoreProxyHandler.RESULT_ERROR)
							.getBytes()));
			response.setStatus(HttpResponseStatus.OK);
		}
		NettyHttpOutput.output(response, event.getChannel());
	}
	
	public RedisCallback(HttpResponse response, MessageEvent event) {
		startTime = System.currentTimeMillis();
		this.response = response;
		this.event = event;
	}
	
	public long getStartTimeMillis() {
		return startTime;
	}
		
	public abstract String getIdentify();
	
}
