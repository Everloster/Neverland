/**
 * CoreProxyHandler.java
 * com.nearme.market.access.server.core
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-9-28 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
 */

package com.nearme.gamecenter.ddz.oauth.handler;

import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.buffer.ChannelBuffers;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.handler.codec.http.HttpMethod;
import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;
import org.jboss.netty.handler.codec.http.HttpResponseStatus;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nearme.gamecenter.ddz.oauth.common.BaseConfig;
import com.nearme.gamecenter.ddz.oauth.server.ClientAccessRequestHandler;
import com.nearme.gamecenter.ddz.oauth.server.NettyHttpOutput;
import com.nearme.gamecenter.ddz.oauth.util.JResultUtil;
import com.nearme.gamecenter.ddz.redis.RedisCallback;
import com.nearme.gamecenter.ddz.redis.RedisServer;
import com.nearme.oauth.model.AccessToken;
import com.nearme.oauth.open.AccountAgent;
import com.nearme.oauth.provider.URLProvider;
import com.nearme.oauth.util.Constants;

/**
 * ClassName:CoreProxyHandler <br>
 * Function: 所有到core的业务处理可以在这里处理 <br>
 * 
 * @author 80036381
 * @version
 * @since Ver 1.1
 * @Date 2012-9-28 下午12:56:50
 */
public class CoreProxyHandler extends ClientAccessRequestHandler {

	private static final Logger logger = LoggerFactory
			.getLogger(CoreProxyHandler.class);

	public static final int RESULT_OK = 1001;
	public static final int RESULT_TIMEOUT = 1002;
	public static final int RESULT_ERROR = 1003;
	public static final int RESULT_INVALID_ID = 1004;

	@Override
	public void handleRequest(HttpRequest request, HttpResponse response,
			MessageEvent e) {
		if (request.getMethod().equals(HttpMethod.POST)) {
			processPost(request, response, e);
		} else {
			processGet(response, e);
		}
	}

	/**
	 * 
	 * @param response
	 * @param e
	 */
	private void processGet(HttpResponse response, MessageEvent e) {
		logger.warn("start process get");
		final ChannelBuffer cb = ChannelBuffers
				.wrappedBuffer("not support get method".getBytes());
		response.setContent(cb);
		response.setStatus(HttpResponseStatus.METHOD_NOT_ALLOWED);
		NettyHttpOutput.output(response, e.getChannel());
	}

	private void processException(HttpResponse response, MessageEvent e) {
		final ChannelBuffer cb = ChannelBuffers.wrappedBuffer("invalid entity"
				.getBytes());
		response.setContent(cb);
		response.setStatus(HttpResponseStatus.NOT_ACCEPTABLE);
		NettyHttpOutput.output(response, e.getChannel());
	}

	/**
	 * 
	 * @param request
	 * @param response
	 * @param e
	 */
	private void processPost(HttpRequest request, HttpResponse response,
			MessageEvent event) {
		final ChannelBuffer cb = request.getContent();
		final String postString = new String(getBytesFromChannelBuffer(cb));
		logger.info("start process post, data : " + postString);
		try {
			final JSONObject jsonObject = new JSONObject(postString);
			final String tokenKey = jsonObject.getString("tokenKey");
			final String tokenSecret = jsonObject.getString("tokenSecret");
			final String oppoId = jsonObject.getString("id");
			final String deviceId = jsonObject.getString("deviceId");
			final int checkResult = checkOppoToken(tokenKey, tokenSecret,
					oppoId);
			switch (checkResult) {
			case RESULT_OK:
				final long startTime = System.currentTimeMillis();
				final String messageId = new StringBuilder((startTime + "_" + jsonObject.hashCode() + "")).toString();
				final RedisCallback callback = new RedisCallback(response, event) {
					
					@Override
					public String getIdentify() {
						return messageId;
					}
				};
				final JSONObject message = new JSONObject();
				message.put("acct_code", "1");
				message.put("acct", oppoId);
				message.put("msg_id", messageId);
				message.put("device_id", deviceId);
				boolean success = RedisServer.getInstance().addMessage(callback, message.toString());
				if (success) {
					// just reutrn , let callback do response.
					return;
				} else {
					response.setContent(ChannelBuffers
							.wrappedBuffer(JResultUtil.getFailureJsonString(RESULT_ERROR)
									.getBytes()));
					response.setStatus(HttpResponseStatus.OK);
				}
				break;
			case RESULT_ERROR:
			case RESULT_INVALID_ID:
			case RESULT_TIMEOUT:
				response.setContent(ChannelBuffers
						.wrappedBuffer(JResultUtil.getFailureJsonString(checkResult)
								.getBytes()));
				response.setStatus(HttpResponseStatus.OK);
				break;
			default:
				break;
			}
			NettyHttpOutput.output(response, event.getChannel());
		} catch (Exception e) {
			logger.warn("receive invalid request post data : " + postString);
			processException(response, event);
		}
	}

	public boolean isAsync() {
		return true;
	}

	/**
	 * 从ChannelBuffer获取byte[]数据
	 * 
	 * @param
	 * @return
	 */
	private byte[] getBytesFromChannelBuffer(ChannelBuffer cb) {
		byte[] data = null;

		int readableBytes = cb.readableBytes();
		if (cb.hasArray() && cb.arrayOffset() == 0
				&& readableBytes == cb.capacity()) {
			data = cb.array();
		} else {
			data = new byte[readableBytes];
			cb.getBytes(0, data, 0, readableBytes);
		}

		return data;
	}

	static {
		// oauth1.0.jar config
		URLProvider.init();
		// config oauth.jar's app key and app secret
		Constants.APP_KEY = BaseConfig.APP_KEY;
		Constants.APP_SECRET = BaseConfig.APP_SECRET;
		// config reuqest timeout(both connection timeout and socket timeout)
		// for access http://thapi.nearme.com.cn/account/GetUserInfoByGame
		Constants.REQUEST_TIMEOUT = 8000;
	}

	private int checkOppoToken(final String tokenKey, final String tokenSecret,
			final String oppoId) {
		final AccessToken mAccessToken = new AccessToken(tokenKey, tokenSecret);
		String data = null;
		try {
			logger.info("start request uc account server");
			data = AccountAgent.getInstance().getGCUserInfo(mAccessToken);
			logger.info("end request uc account server");
			final JSONObject userInfo = new JSONObject(data)
					.getJSONObject("BriefUser");
			if (userInfo.getString("id").equals(String.valueOf(oppoId))) {
				logger.info(
						"check oppo token success, tokenKey:{}, tokenSecret:{}, oppoId:{}.",
						tokenKey, tokenSecret, oppoId);
				return RESULT_OK;
			} else {
				logger.error("check token id invalid ,"
						+ "tokenKey:{} tokenSecret:{} oppoId:{}.", tokenKey,
						tokenSecret, oppoId);
				return RESULT_INVALID_ID;
			}
		} catch (Exception e) {
			if (data != null) {
				logger.error(
						"check token error, get oppo userinfo: "
								+ e.getMessage() + ", userinfo:{}", data);
			}
			logger.error("check token error : " + e.getMessage()
					+ ", tokenKey:{} tokenSecret:{} oppoId:{}.", tokenKey,
					tokenSecret, oppoId);
			return RESULT_ERROR;
		}
	}
}
