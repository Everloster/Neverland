/**
 * BusinessServerFactory.java
 * com.nearme.base.netty.connect
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-12-7 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.connect;

import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicInteger;

import com.nearme.base.netty.client.AbstractNettyBusinessClientFactory;
import com.nearme.base.netty.client.INettyClient;

/**
 * ClassName:BusinessServerFactory <br>
 * Function: 业务服务器连接分配 <br>
 * 通过select方法从多个server端中选择,默认的为平均分配<br>
 * 重写select方法来实现不同的选择方式,<br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-12-7  上午10:53:50
 */
public abstract class AbstractBusinessServerFactory<T, D> implements IBusinessServerFactory<T, D> {
	private AbstractNettyBusinessClientFactory<T, D> factory;
	private int clientCount;	//每个服务建立的连接个数
	private int timeout;		//建立连接时的超时时间
	private AtomicInteger position = new AtomicInteger();
	private Executor executor;

	/**
	 *
	 * Creates a new instance of AbstractBusinessServerFactory.
	 *
	 * @param factory 生成连接的工厂类
	 */
	public AbstractBusinessServerFactory(AbstractNettyBusinessClientFactory<T, D> factory, Executor executor) {
		this(factory, DEFAULT_CLIENT_TIMEOUT, DEFAULT_CLIENT_COUNT, executor);
	}

	/**
	 *
	 * Creates a new instance of AbstractBusinessServerFactory.
	 *
	 * @param factory 生成连接的工厂类
	 * @param timeout 建立连接的超时时间
	 * @param clientCount 每个服务建立的连接个数
	 */
	public AbstractBusinessServerFactory(AbstractNettyBusinessClientFactory<T, D> factory, int timeout, int clientCount) {
		this(factory, timeout, clientCount, null);
	}

	/**
	 *
	 * Creates a new instance of AbstractBusinessServerFactory.
	 *
	 * @param factory 生成连接的工厂类
	 * @param timeout 建立连接的超时时间
	 * @param clientCount 每个服务建立的连接个数
	 */
	public AbstractBusinessServerFactory(AbstractNettyBusinessClientFactory<T, D> factory, int timeout, int clientCount, Executor executor) {
		this.factory = factory;
		this.clientCount = clientCount <= 0 ? DEFAULT_CLIENT_COUNT : clientCount;
		this.timeout = timeout <= 0 ? DEFAULT_CLIENT_TIMEOUT : timeout;
		this.executor = executor;
	}

	/**
	 * 获取连接核心业务服务的客户端
	 * @param
	 * @return
	 */
	@Override
	public INettyClient<T, D> getConnectClient() throws Exception {
		BusinessServerInfo[] servers = loadServerInfo();
		if(null == servers) {
			return null;
		}

		BusinessServerInfo selectServer = null;
		if(servers.length == 1) {
			selectServer = servers[0];
		} else {
			selectServer = select(servers);
		}

		return factory.get(
				selectServer.getServerIp(), 	//服务器ip
				selectServer.getServerPort(), 	//服务器端口
				timeout, 		//创建客户端的超时时间
				clientCount,	//创建客户端的个数
				executor);
	}

	/**
	 * 获取核心业务服务的所有客户端连接
	 * @param
	 * @return
	 */
	@Override
	public INettyClient<T, D>[] getConnectClients() throws Exception {
		BusinessServerInfo[] servers = loadServerInfo();
		if(null == servers) {
			return null;
		}

		//连接的服务器
		int serverCount = servers.length;
		INettyClient<T, D>[] clients = new INettyClient[serverCount];
		for(int i = 0; i < serverCount; i++) {
			BusinessServerInfo selectServer = servers[i];
			clients[i] = factory.get(
					selectServer.getServerIp(), 	//服务器ip
					selectServer.getServerPort(), 	//服务器端口
					timeout, 		//创建客户端的超时时间
					clientCount,	//创建客户端的个数
					executor);
		}

		return clients;
	}

	/**
	 * 从多个服务器中选择一个,默认按照平均分配的方式
	 * @param
	 * @return
	 */
	@Override
	public BusinessServerInfo select(BusinessServerInfo[] servers) {
		int currentPos = position.getAndIncrement();
		//防止溢出
		if(currentPos < 0) {
			currentPos = 0;
			position.set(1);
		}

		return servers[currentPos % servers.length];
	}
}

