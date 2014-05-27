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

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import org.jboss.netty.bootstrap.Bootstrap;
import org.jboss.netty.channel.ChannelFuture;
import org.jboss.netty.channel.ChannelFutureListener;
import org.jboss.netty.util.Timeout;
import org.jboss.netty.util.Timer;
import org.jboss.netty.util.TimerTask;

import com.nearme.base.netty.common.NettyConstants;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.RequestData;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ResponseData;
import com.nearme.base.stat.ExecutorStat;
import com.nearme.base.stat.ExecutorStatPool;
import com.oppo.base.common.OConstants;

/**
 * ClassName:NettyBusinessClient <br>
 * Function: 与netty搭建的业务层server端通讯的客户端，异步接收数据 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-9-1  上午11:13:21
 */
public class NettyBusinessAsyncClient extends AbstractNettyBusinessClient<RequestData, ResponseData> {

	private Timer timer;

	protected ConcurrentHashMap<Long, HttpInstanceInfo> responseMap =
		new ConcurrentHashMap<Long, HttpInstanceInfo>(16, 0.75f, 64);

	private boolean clientRunning = true;

	public NettyBusinessAsyncClient(Bootstrap bootstrap, ChannelFuture cf, String identifier, Timer timer) {
		super(bootstrap, cf, identifier);
		this.timer = timer;
//
//		initWatchThread();
	}

//	/**
//	 * 初始化监控线程
//	 * @param
//	 * @return
//	 */
//	private void initWatchThread() {
//		Thread th = new Thread(new Runnable(){
//			@Override
//			public void run() {
//				while(threadStart) {
//					Collection<HttpInstanceInfo> values = responseMap.values();
//					for(HttpInstanceInfo httpInstance : values) {
//						RequestData request = httpInstance.getRequestData();
//						long time = System.currentTimeMillis() - httpInstance.getStartTime();
//						if(time > request.getTimeout()) {
//							//超时移除
//							values.remove(httpInstance);
//
//							//通知超时
//							ResponseData.Builder builder = ResponseData.newBuilder();
//							builder.setRequestId(request.getId());
//							builder.setCode(NettyConstants.REQUEST_TIMEOUT);
//							builder.setMessage("数据发送超时,time:" + time + "ms,id:" + request.getId());
//							builder.setResponseType(request.getRequestType());
//							builder.setTimeout(request.getTimeout());
//							//执行失败
//							ExecutorStat stat = ExecutorStatPool.getInstance().getExecutorStat(getExecutorStatIdentifier(request.getRequestType()));
//							stat.addStat(false, System.currentTimeMillis() - httpInstance.getStartTime());
//							httpInstance.onResponse(builder.build());
//						}
//					}
//
//					try {
//						//每10秒扫描一次
//						Thread.sleep(10000);
//					} catch(Exception ex) {
//					}
//				}
//			}
//		});
//		th.setName("Thread-timeout-watch-" + identifier + "-" + th.getId());
//		th.setDaemon(true);
//		th.setPriority(Thread.MIN_PRIORITY);
//		th.start();
//	}

	/**
	 * 从服务端获取数据
	 * @param
	 * @return
	 */
	@Override
	public ResponseData getResponse(RequestData request) throws Exception {
		long startTime = System.currentTimeMillis();

		ExecutorStat stat = ExecutorStatPool.getInstance().getExecutorStat(getExecutorStatIdentifier(request.getRequestType()));
		//发送请求数据
		try {
			sendRequest(request, request.getTimeout());

			//执行成功时的统计
			stat.addStat(true, System.currentTimeMillis() - startTime);
		} catch(Exception ex) {
			//执行错误时的统计
			stat.addStat(false, System.currentTimeMillis() - startTime);
		}

		return null;
	}

	/**
	 * 从服务端获取数据
	 * @param
	 * @return
	 */
	@Override
	public ResponseData getResponse(RequestData request, Object addtional) throws Exception {
		if(!(addtional instanceof HttpInstanceInfo)) {
			throw new IllegalArgumentException("wrong argument:" + (null == addtional ? "null" : addtional.getClass().getName()));
		}

		HttpInstanceInfo httpInstance = (HttpInstanceInfo)addtional;
		//如果没有设置RequestData,则设置,方便进行监控
		if(null == httpInstance.getRequestData()) {
			httpInstance.setRequestData(request);
		}

		try {
			responseMap.put(request.getId(), httpInstance);

			//发送请求数据
			sendRequest(request, request.getTimeout());

			//超时检查任务
			Timeout timeout = timer.newTimeout(new RequestTimeoutTask(httpInstance), request.getTimeout(), TimeUnit.MILLISECONDS);
			if (httpInstance.isRequestComplete()) {
				timeout.cancel();
			} else {
				httpInstance.setTimeout(timeout);
			}

			//执行成功时的统计在putResponseData中处理,这里统计发送部分的时间
			ExecutorStat stat = ExecutorStatPool.getInstance().getExecutorStat("send request");
			stat.addStat(true, System.currentTimeMillis() - httpInstance.getStartTime());
		} catch(Exception ex) {
			//执行错误时的统计
			ExecutorStat stat = ExecutorStatPool.getInstance().getExecutorStat(getExecutorStatIdentifier(request.getRequestType()));
			stat.addStat(false, System.currentTimeMillis() - httpInstance.getStartTime());

			throw ex;
		}

		return null;
	}

	/**
	 * 接收返回信息
	 */
	@Override
	public void putResponse(ResponseData response) throws Exception {
		HttpInstanceInfo httpInstance = responseMap.remove(response.getRequestId());
		if(null == httpInstance) {
			//可能是超时被移除了，这里不再输出日志了，防止重复输出，误导分析师
			this.warn("request id not found,may be it was output already. id:{},requestType:{},code:{},msg:{}", new Object[] {
					response.getRequestId(), response.getResponseType(), response.getCode(), response.getMessage()});
			return;
		}

		Timeout timeout = httpInstance.getTimeout();
		if (null != timeout && !timeout.isExpired()) {
			timeout.cancel();	//取消超时判断的任务
		}

		//执行时间统计
		boolean isSucc = response.getCode() >= NettyConstants.RESPONSE_NORMAL;	//是否执行成功
		ExecutorStat stat = ExecutorStatPool.getInstance().getExecutorStat(getExecutorStatIdentifier(response.getResponseType()));
		stat.addStat(isSucc, System.currentTimeMillis() - httpInstance.getStartTime());
		if(!isSucc) {
			this.warn("request error,id:{},requestType:{},code:{},msg:{}", new Object[] {
					response.getRequestId(), response.getResponseType(), response.getCode(), response.getMessage()});
		}

		//事件方式处理结果数据
		httpInstance.onResponse(response);
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

		final NettyBusinessAsyncClient thisClient = this;
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
					long total = System.currentTimeMillis() - startTime;
					if (total >= timeout) {
						errorMsg = "数据发送超时,time:" + total + "ms,id:" + request.getId();
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
					//移除未连接的client
					NettyBusinessClientFactory.getInstance().removeClient(
							identifier, thisClient, "channel is not connected(async).");
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

		this.clientRunning = false;
		this.responseMap.clear();
	}

	private String getExecutorStatIdentifier(int requestType) {
		return "requestType-" + requestType;
	}

	/**
	 * 请求超时任务处理
	 * @author   80036381
	 * @version  NettyBusinessAsyncClient
	 * @since    Ver 1.1
	 * @Date	 2013-3-15  上午9:51:43
	 * @see
	 */
	private final class RequestTimeoutTask implements TimerTask {
		private HttpInstanceInfo httpInstance;

		public RequestTimeoutTask(HttpInstanceInfo httpInstance) {
			this.httpInstance = httpInstance;
		}

		@Override
		public void run(Timeout timeout) throws Exception {
			if (httpInstance.isRequestComplete()
					|| timeout.isCancelled()
					|| !clientRunning) {
                return;
            }

			//执行时间
			long executeTime = System.currentTimeMillis() - httpInstance.getStartTime();

			RequestData request = httpInstance.getRequestData();
			long settingTimeout = request.getTimeout();
			if(settingTimeout <= executeTime) {
				//通知超时
				ResponseData.Builder builder = ResponseData.newBuilder();
				builder.setRequestId(request.getId());
				builder.setCode(NettyConstants.REQUEST_TIMEOUT);
				builder.setMessage("数据读取超时,time(ms):" + executeTime);
				builder.setResponseType(request.getRequestType());
				builder.setTimeout(request.getTimeout());

				putResponse(builder.build());
			} else {
				//还未超时则重新处理
				timer.newTimeout(this, settingTimeout - settingTimeout, TimeUnit.MILLISECONDS);
			}
		}
	}
}

