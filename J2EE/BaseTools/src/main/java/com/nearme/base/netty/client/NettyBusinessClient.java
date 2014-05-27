/**
 * NettyBusinessClient.java
 * com.nearme.base.netty.client
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-9-1 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.client;

import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import org.jboss.netty.bootstrap.Bootstrap;
import org.jboss.netty.channel.ChannelFuture;
import org.jboss.netty.channel.ChannelFutureListener;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.handler.codec.http.HttpRequest;

import com.nearme.base.netty.common.DataHelper;
import com.nearme.base.netty.common.NettyConstants;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ProtobufResponse;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.RequestData;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ResponseData;
import com.nearme.base.stat.ExecutorStat;
import com.nearme.base.stat.ExecutorStatPool;
import com.oppo.base.common.OConstants;

/**
 * ClassName:NettyBusinessClient <br>
 * Function: 与netty搭建的业务层server端通讯的客户端，等待式接收数据 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-9-1  上午11:13:21
 */
public class NettyBusinessClient extends AbstractNettyBusinessClient<RequestData, ResponseData> {
	protected ConcurrentHashMap<Long, ArrayBlockingQueue<Object>> responseMap =
		new ConcurrentHashMap<Long, ArrayBlockingQueue<Object>>(16, 0.75f, 64);

	public NettyBusinessClient(Bootstrap bootstrap, ChannelFuture cf, String identifier) {
		super(bootstrap, cf, identifier);
	}

	/**
	 * 从服务端获取数据
	 * @param requestType 请求类型，根据此类型查找对应的处理对象
	 * @param httpRequest
	 * @param e
	 * @param timeout 超时时间
	 * @return 服务端的返回数据
	 */
	public ResponseData getResponse(int requestType, HttpRequest httpRequest, MessageEvent e,
			int timeout) throws Exception {
		RequestData request = DataHelper.getRequestData(requestType, httpRequest, e, timeout);

		return this.getResponse(request);
	}

	/**
	 * 从服务端获取数据
	 * @param requestType 请求类型，根据此类型查找对应的处理对象
	 * @param data 请求数据
	 * @param headers 转发客户端的头
	 * @param clientIp 客户端IP
	 * @param timeout 超时时间
	 * @return 服务端的返回数据
	 */
	public ResponseData getResponse(int requestType, byte[] data,
			Map<String, String> headers, String clientIp, int timeout)
		throws Exception {
		RequestData request = DataHelper.getRequestData(requestType, data, headers, clientIp, timeout);

		return getResponse(request, null);
	}

	/**
	 * 从服务端获取数据
	 * @param
	 * @return
	 */
	@Override
	public ResponseData getResponse(RequestData request) throws Exception {
		return getResponse(request, null);
	}

	/**
	 * 从服务端获取数据
	 * @param
	 * @return
	 */
	public ResponseData getResponse(RequestData request, Object addtional) throws Exception {
		long beginTime = System.currentTimeMillis();
		ArrayBlockingQueue<Object> responseQueue = new ArrayBlockingQueue<Object>(1);
		responseMap.put(request.getId(), responseQueue);

		//统计执行时间
		ExecutorStat stat = ExecutorStatPool.getInstance().getExecutorStat("requestType-" + request.getRequestType());

		//发送请求数据
		Object result = null;
		try {
			sendRequest(request, request.getTimeout());
		} catch (Exception e) {
			responseMap.remove(request.getId());
			responseQueue = null;

			//执行错误时的统计
			stat.addStat(false, System.currentTimeMillis() - beginTime);
			throw e;
		}

		//获取返回结果
		try {
			result = responseQueue.poll(
					request.getTimeout() - (System.currentTimeMillis() - beginTime),
					TimeUnit.MILLISECONDS);
		} catch(InterruptedException e){
			responseMap.remove(request.getId());

			//执行错误时的统计
			stat.addStat(false, System.currentTimeMillis() - beginTime);
			throw new Exception("获取返回数据失败", e);
		}

		responseMap.remove(request.getId());

		ResponseData response = null;
		if(result instanceof ResponseData){
			response = (ResponseData) result;
		} else if(result instanceof ProtobufResponse){
			ProtobufResponse responses = (ProtobufResponse) result;
			int size = responses.getResponseListCount();
			for(int i = 0; i < size; i++) {
				ResponseData tmpResponse = responses.getResponseList(i);
				if(tmpResponse.getRequestId() == request.getId()) {
					response = tmpResponse;
				} else {
					putResponse(tmpResponse);
				}
			}
		} else{
			//执行错误时的统计
			stat.addStat(false, System.currentTimeMillis() - beginTime);
			throw new Exception("只支持ResponseData或ProtobufResponse:" + ((null == result) ? "null" : result.getClass().getSimpleName()));
		}

		if (null == response) {
			String errorMsg = "获取返回数据超时:"
					+ request.getTimeout() + " ms,"
					+ getServerIP() + ":" + getServerPort()
					+ ",id:" + request.getId();

			//执行错误时的统计
			stat.addStat(false, System.currentTimeMillis() - beginTime);
			throw new Exception(errorMsg);
		}

		//负数表示发生错误
		if (response.getCode() < NettyConstants.RESPONSE_NORMAL) {
			String errorMsg = "请求错误: " + getServerIP() + ":" + getServerPort()
				+ ",id:" + request.getId()
				+ ",code:" + response.getCode()
				+ ",msg:" + response.getMessage();

			//执行错误时的统计
			stat.addStat(false, System.currentTimeMillis() - beginTime);

			throw new Exception(errorMsg);
		}

		//执行成功时的统计
		stat.addStat(true, System.currentTimeMillis() - beginTime);

		return response;
	}

	/**
	 * 接收返回信息
	 */
	public void putResponse(ResponseData response) throws Exception {
		if (!responseMap.containsKey(response.getRequestId())) {
			warn("id:" + response.getRequestId() + ",未找到对应请求!");
			return;
		}

		try {
			ArrayBlockingQueue<Object> queue = responseMap.get(response.getRequestId());
			if (null != queue) {
				queue.put(response);
			} else {
				//没找到对应的回传节点
			}
		} catch (InterruptedException e) {
			error("id:" + response.getRequestId(), e);
		}
	}

	/**
	 * 发送请求
	 * @param request
	 * @param timeout
	 * @return
	 */
	protected void sendRequest(final RequestData request, final int timeout) {
		final long startTime = System.currentTimeMillis();
		ChannelFuture writeFuture = cf.getChannel().write(request);

		final INettyClient<RequestData, ResponseData> thisClient = this;
		writeFuture.addListener(new ChannelFutureListener() {
			public void operationComplete(ChannelFuture future)
					throws Exception {
				if (future.isSuccess()) {
					return;
				}

				String errorMsg = OConstants.EMPTY_STRING;
				int errorCode = NettyConstants.REQUEST_UNKNOWN_ERROR;
				Throwable throwable = null;
				if (future.isCancelled()) {
					//请求中断
					errorMsg = "发送到 " + cf.getChannel().toString() + " 的请求被中断,id:" + request.getId();
					errorCode = NettyConstants.REQUEST_CANCELLED;
					throwable = new Exception("请求被中断");
				} else {
					//发送请求超时
					long executeTime = System.currentTimeMillis() - startTime;
					if (executeTime >= timeout) {
						errorMsg = "数据发送超时,time:" + executeTime + "ms,id:" + request.getId();
						errorCode = NettyConstants.REQUEST_TIMEOUT;
						throwable = new Exception("数据发送超时");
					} else {
						errorMsg = "发送数据时发生错误" + cf.getChannel().toString();
						errorCode = NettyConstants.REQUEST_NOT_SUCCESS;
						throwable = future.getCause();
					}
				}

				if (cf.getChannel().isConnected()) {
					cf.getChannel().close();
				} else {
					NettyBusinessClientFactory.getInstance().removeClient(
							identifier, thisClient, "channel is not connected(sync).");
				}

				error(errorMsg, throwable);

				//构造一个返回的实体
				ResponseData.Builder builder = ResponseData.newBuilder();
				builder.setRequestId(request.getId());
				builder.setCode(errorCode);
				builder.setMessage(errorMsg);
				builder.setResponseType(request.getRequestType());
				builder.setTimeout(request.getTimeout());

				thisClient.putResponse(builder.build());
			}
		});
	}

	/**
	 * 停止监控
	 * @param
	 * @return
	 * @Override
	 */
	public void stop() {
		super.stop();

		responseMap.clear();
	}
}

