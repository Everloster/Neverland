/**
 * NettyHttpManager.java
 * com.nearme.base.netty.http
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-3-29 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.http;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;

import org.jboss.netty.handler.codec.http.HttpRequest;
import org.jboss.netty.handler.codec.http.HttpResponse;

import com.nearme.base.concurrent.ExecutorManager;

/**
 * ClassName:NettyHttpManager <br>
 * Function: netty http管理器 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-3-29  上午9:00:26
 */
public class NettyHttpManager {

//	/**
//     * 最大连接数
//     */
//    public final static int MAX_TOTAL_CONNECTIONS = 1024;
//
//    /**
//     * 每个路由最大连接数
//     */
//    public final static int MAX_ROUTE_CONNECTIONS = 512;

    private INettyHttpRequest httpRequest;
    private ExecutorService executorService;
    //private ClientBootstrap bootstrap;
	private ConcurrentHashMap<String, NettyHttpPool> poolMap;
	private Map<String, Object> optionMap;

	/**
	 *
	 */
	public NettyHttpManager(INettyHttpRequest httpRequest, ExecutorService executor) {
		this.httpRequest = httpRequest;
		this.executorService = executor;

		poolMap = new ConcurrentHashMap<String, NettyHttpPool>();
		optionMap = new HashMap<String, Object>();
	}

	public ChannelContext getChannelAndRequest(ChannelContext context, HttpRequest request) {
		NettyHttpPool httpPool = getHttpPool(context);
		return httpPool.getChannelAndRequest(context, request);
	}

	/**
	 * 释放连接-根据返回数据及池状况判断是否加入池中
	 * @param
	 * @return
	 */
	public void release(ChannelContext context, HttpResponse response) {
		NettyHttpPool httpPool = getHttpPool(context);
		httpPool.release(context, response);
	}

	/**
	 * 释放连接-直接关闭
	 * @param
	 * @return
	 */
	public void releaseDirect(ChannelContext context) {
		NettyHttpPool httpPool = getHttpPool(context);
		httpPool.releaseDirect(context);
	}

	protected NettyHttpPool getHttpPool(ChannelContext context) {
		String key = context.getIdentifier();
		NettyHttpPool pool = poolMap.get(key);
		if(null == pool) {
			pool = new NettyHttpPool(executorService, httpRequest, optionMap);
			NettyHttpPool oldPool = poolMap.putIfAbsent(key, pool);
			if(null != oldPool) {
				pool.close();
				pool = oldPool;
			}
		}

		return pool;
	}

	/**
	 * 关闭池
	 * @param
	 * @return
	 */
	public void close() {
		//关闭所有的Channel
		for(NettyHttpPool pool : poolMap.values()) {
			pool.close();
		}

		poolMap.clear();
		ExecutorManager.shutdown(executorService);
	}

	/**
	 * 设置ClientBootstap的选项
	 * @param
	 * @return
	 */
	public void setBootstrapOption(String key, Object value) {
		if(null == value) {
			optionMap.remove(key);
		} else {
			optionMap.put(key, value);
		}
	}
}

