/**
 * NettyBusinessClientFactory.java
 * com.nearme.base.netty.client
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-5-4 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.client;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.FutureTask;

import org.jboss.netty.bootstrap.Bootstrap;
import org.jboss.netty.channel.ChannelFuture;

import com.nearme.base.netty.common.ProtobufWrapperProtocol.RequestData;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ResponseData;

/**
 * ClassName:NettyBusinessClientFactory <br>
 * Function: 与netty搭建的业务层server端通讯的客户端生成工厂 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-5-4  下午04:38:43
 */
public class NettyBusinessClientFactory extends AbstractNettyBusinessClientFactory<RequestData, ResponseData> {

	private static ConcurrentHashMap<String, FutureTask<List<INettyClient<RequestData, ResponseData>>>> clientMap =
		new ConcurrentHashMap<String, FutureTask<List<INettyClient<RequestData, ResponseData>>>>();

	private static NettyBusinessClientFactory instance = new NettyBusinessClientFactory();

	private NettyBusinessClientFactory() {
		super();
	}

	public static NettyBusinessClientFactory getInstance() {
		return instance;
	}

	@Override
	protected AbstractNettyBusinessClientHandler<RequestData, ResponseData> getClientHandler(
			AbstractNettyBusinessClientFactory<RequestData, ResponseData> clientFactory,
			String key,
			Executor executor) {
		return new NettyBusinessClientHandler(clientFactory, key, executor);
	}

	@Override
	protected INettyClient<RequestData, ResponseData> getNettyClient(Bootstrap bootstrap, ChannelFuture future, String key) {
		return new NettyBusinessClient(bootstrap, future, key);
	}

	@Override
	protected ConcurrentHashMap<String, FutureTask<List<INettyClient<RequestData, ResponseData>>>> getClientMap() {
		return clientMap;
	}
}

