/**
 * AbstractNettyBusinessClientFactory.java
 * com.nearme.base.netty.client
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-9-28 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.client;

import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.FutureTask;
import java.util.concurrent.atomic.AtomicInteger;

import org.jboss.netty.bootstrap.Bootstrap;
import org.jboss.netty.bootstrap.ClientBootstrap;
import org.jboss.netty.channel.ChannelFuture;
import org.jboss.netty.channel.ChannelHandler;
import org.jboss.netty.channel.ChannelPipeline;
import org.jboss.netty.channel.ChannelPipelineFactory;
import org.jboss.netty.channel.Channels;
import org.jboss.netty.channel.socket.nio.NioClientSocketChannelFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nearme.base.concurrent.ExecutorManager;
import com.nearme.base.netty.codec.ProtobufInOneDecoder;
import com.nearme.base.netty.codec.ProtobufInOneEncoder;
import com.nearme.base.netty.common.NettyConstants;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ResponseData;

/**
 * ClassName:AbstractNettyBusinessClientFactory <br>
 * Function: 与netty搭建的业务层server端通讯的客户端生成工厂 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-9-28  上午09:54:26
 */
public abstract class AbstractNettyBusinessClientFactory<T, D> {
	/**
	 * 客户端tcpNoDelelay属性名
	 */
	public static final String CLIENT_TCP_NODELAY = "com.nearme.base.netty.client.tcpNoDelay";

	/**
	 * 客户端reuseAddress属性名
	 */
	public static final String CLIENT_REUSE_ADDRESS = "com.nearme.base.netty.client.reuseAddress";

	/**
	 * 客户端keepAlive属性名
	 */
	public static final String CLIENT_KEEP_ALIVE = "com.nearme.base.netty.client.keepAlive";

	/**
	 * 客户端tcpNoDelay属性默认值
	 */
	public static final String DEFAULT_CLIENT_TCP_NODELAY = "true";

	/**
	 * 客户端reuseAddress属性默认值
	 */
	public static final String DEFAULT_CLIENT_REUSE_ADDRESS = "true";

	private Random rand = new Random();

//	private ClientBootstrap bootstrap = null;
	private Logger logger = LoggerFactory.getLogger(this.getClass());
	private AtomicInteger clientIndex = new AtomicInteger();
	private ExecutorService es = ExecutorManager.newFixedThreadPool(8);

	protected AbstractNettyBusinessClientFactory() {
//		bootstrap = createClientBootstrap();
	}

	protected ClientBootstrap createClientBootstrap() {
		int currentIndex = clientIndex.incrementAndGet();
		Executor bossExecutor = ExecutorManager.newCachedThreadPool("Client-Boss-Executor-" + currentIndex);
		Executor workerExecutor = ExecutorManager.newCachedThreadPool("Client-Worker-Executor-" + currentIndex);
		ClientBootstrap bootstrap = new ClientBootstrap(new NioClientSocketChannelFactory(bossExecutor, workerExecutor));
		bootstrap.setOption("tcpNoDelay", Boolean.parseBoolean(
				System.getProperty(CLIENT_TCP_NODELAY, DEFAULT_CLIENT_TCP_NODELAY)));
		bootstrap.setOption("reuseAddress", Boolean.parseBoolean(
				System.getProperty(CLIENT_REUSE_ADDRESS, DEFAULT_CLIENT_REUSE_ADDRESS)));

		String keepAlive = System.getProperty(CLIENT_KEEP_ALIVE);
		if(null != keepAlive) {
			bootstrap.setOption("keepAlive", Boolean.parseBoolean(keepAlive));
		}

		return bootstrap;
	}

	/**
	 * 获取一个客户端连接
	 * @param
	 * @return
	 */
	public INettyClient<T, D> get(final String serverIp, final int serverPort,
			final int timeout, final int clientCount, Executor executor)
			throws Exception {
		return get(null, serverIp, serverPort, timeout, clientCount, executor);
	}

	/**
	 * 获取一个客户端连接
	 * @param
	 * @return
	 */
	public INettyClient<T, D> get(String identifier, final String serverIp, final int serverPort,
			final int timeout, final int clientCount, final Executor executor)
			throws Exception {
		if(null == identifier) {
			identifier = serverIp + ':' + serverPort;
		}

		FutureTask<List<INettyClient<T, D>>> task = getClientMap().get(identifier);
		if(null != task) {
			return getClientFromList(task.get(), identifier);
		} else {
			final String clientKey = identifier;
			FutureTask<List<INettyClient<T, D>>> newTask = new FutureTask<List<INettyClient<T, D>>>(
					new Callable<List<INettyClient<T, D>>>() {
						public List<INettyClient<T, D>> call() throws Exception {
							//创建指定个数的客户端连接
							List<INettyClient<T, D>> clients = new ArrayList<INettyClient<T, D>>(clientCount);
							for (int i = 0; i < clientCount; i++) {
								INettyClient<T, D> client = createClient(serverIp, serverPort, timeout, clientKey, executor);
								if (null != client) {
									clients.add(client);
								} else {
									// 出错则停止后面的操作
									break;
								}
							}

							return clients;
						}
					});

			FutureTask<List<INettyClient<T, D>>> currentTask = getClientMap().putIfAbsent(identifier, newTask);
			if(currentTask == null) {
				//不存在则运行任务
				newTask.run();
			} else {
				newTask = currentTask;
			}

			return getClientFromList(newTask.get(), identifier);
		}
	}

	/**
	 * 移除使用中的client
	 * @param
	 * @return
	 */
	public void removeClient(final String identifier, INettyClient<T, D> client, final String reason) {
		FutureTask<List<INettyClient<T, D>>> task = getClientMap().remove(identifier);
		if(null != task) {
			try {
				if (!task.isDone()) {
//					task.cancel(true);
					return;
				}

				final long startMillis = System.currentTimeMillis();
				final List<INettyClient<T, D>> clientList = task.get();
				es.execute(new Runnable() {
					@Override
					public void run() {
						for(int i = 0, size = clientList.size(); i < size; i++) {
							//停止监控服务
							clientList.get(i).stop();
						}

						long totalMillis = System.currentTimeMillis() - startMillis;
						logger.info("{}被移除:{},耗时:{}ms", identifier, reason, totalMillis);
					}
				});

			} catch (Exception e) {
				logger.error(identifier + "移除时发生错误:" + reason, e);
				return;
			}
		}
	}

	public void shutdown() {
		ExecutorManager.shutdown(es);

		for (String key : getClientMap().keySet()) {
			removeClient(key, null, "shutdown");
		}
	}

	/**
	 * 从指定列表中获取一个连接
	 * @param
	 * @return
	 */
	protected INettyClient<T, D> getClientFromList(List<INettyClient<T, D>> clientList, String identifier) {
		int clientCount = null == clientList ? 0 : clientList.size();

		if (clientCount == 0) {
			getClientMap().remove(identifier);
			return null;
		} else if (clientCount == 1) {
			return clientList.get(0);
		} else {
			//从多个客户端连接中随机取出一个
			return clientList.get(rand.nextInt(clientCount));
		}
	}

	/**
	 * 创建一个指定ip及端口的连接
	 * @param
	 * @return
	 */
	protected INettyClient<T, D> createClient(String serverIp, int serverPort,
			int timeout, String key, Executor executor) throws Exception {
		ClientBootstrap bootstrap = createClientBootstrap();
		if(timeout < NettyConstants.MIN_CLIENT_TIMEOUT) {
			bootstrap.setOption("connectTimeoutMillis", NettyConstants.MIN_CLIENT_TIMEOUT);
		} else{
			bootstrap.setOption("connectTimeoutMillis", timeout);
		}

		AbstractNettyBusinessClientHandler<T, D> handler = getClientHandler(this, key, executor);
		bootstrap.setPipelineFactory(getChannelPipelineFactory(handler));
		ChannelFuture future = bootstrap.connect(new InetSocketAddress(serverIp, serverPort));
		future.awaitUninterruptibly(timeout);

		Exception exception = null;
		if (!future.isDone()) {
			exception = new Exception("创建连接超时 " + serverIp + ":" + serverPort);
		}

		if (future.isCancelled()) {
			exception = new Exception("连接被取消 " + serverIp + ":" + serverPort);
		}

		if (!future.isSuccess()) {
			exception = new Exception("创建连接出错 " + serverIp + ":" + serverPort, future.getCause());
		}

		if (null != exception) {
			bootstrap.releaseExternalResources();
			logger.error("连接至" + serverIp + ":" + serverPort + "失败。", exception);

			return null;
		}

		INettyClient<T, D> client = getNettyClient(bootstrap, future, key);
		handler.setClient(client);
		return client;
	}

	/**
	 * 通过指定的ChannelHandler创建ChannelPipelineFactory(默认使用pb通信)
	 * @param
	 * @return
	 */
	protected ChannelPipelineFactory getChannelPipelineFactory(final ChannelHandler handler) {
		return new ChannelPipelineFactory() {
			@Override
			public ChannelPipeline getPipeline() throws Exception {
				ChannelPipeline pipeline = Channels.pipeline();
				pipeline.addLast("pbInOneDecoder", new ProtobufInOneDecoder(ResponseData.getDefaultInstance()));
				pipeline.addLast("pbInOneEncoder", new ProtobufInOneEncoder());
				pipeline.addLast("handler", handler);
				return pipeline;
			}
		};
	}

//	public ClientBootstrap getClientBootstrap() {
//		return bootstrap;
//	}

	/**
	 * 获取数据处理器
	 * @param
	 * @return
	 */
	protected abstract AbstractNettyBusinessClientHandler<T, D> getClientHandler(
			AbstractNettyBusinessClientFactory<T, D> clientFactory, String key, Executor executor);

	/**
	 * 获取用于存放client的map
	 * @param
	 * @return
	 */
	protected abstract ConcurrentHashMap<String, FutureTask<List<INettyClient<T, D>>>> getClientMap();

	/**
	 * 获取一个client对象
	 * @param
	 * @return
	 */
	protected abstract INettyClient<T, D> getNettyClient(Bootstrap bootstrap, ChannelFuture future, String key);
}

