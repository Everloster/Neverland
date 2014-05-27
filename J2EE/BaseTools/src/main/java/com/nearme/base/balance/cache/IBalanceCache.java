/**
 * IBalanceCache.java
 * com.nearme.base.balance
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-28 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance.cache;

import java.util.List;

import com.nearme.base.balance.model.ServerInfo;

/**
 * ClassName:IBalanceCache <br>
 * Function: 负载均衡的缓存 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-28  上午9:31:39
 */
public interface IBalanceCache {
	/**
	 * 根据缓存标识获取缓存的服务信息
	 * @param groupId 组id
	 * @param cacheIdentifier 缓存标识
	 * @return
	 */
	ServerInfo getCacheServer(String groupId, String cacheIdentifier);

	/**
	 * 获取及时的连接服务信息
	 * @param groupId 组id
	 * @param cacheIdentifier 缓存标识
	 * @param servers 备选的服务器
	 * @return
	 */
	ServerInfo getConnectServer(String groupId, String cacheIdentifier, List<ServerInfo> servers);

	/**
	 * 将服务信息存入指定标识的缓存
	 * @param groupId 组id
	 * @param cacheIdentifier 缓存标识
	 * @param serverInfo
	 * @return
	 */
	void putServerToCache(String groupId, String cacheIdentifier, ServerInfo serverInfo);
}

