/**
 * HashBalanceCache.java
 * com.nearme.base.balance.cache
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-6-17 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance.cache;

import java.util.List;

import com.nearme.base.balance.model.ServerInfo;

/**
 * ClassName:HashBalanceCache <br>
 * Function: 根据标记的hash值分配 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-6-17  上午9:54:14
 */
public class HashBalanceCache extends DefaultBalanceCache {

	/**
	 * hash方式不需要缓存
	 * @see com.nearme.base.balance.cache.DefaultBalanceCache#getCacheServer(java.lang.String)
	 */
	@Override
	public ServerInfo getCacheServer(String groupId, String cacheIdentifier) {
		return null;
	}

	@Override
	public ServerInfo getConnectServer(String groupId, String cacheIdentifier, List<ServerInfo> servers) {
		int size = null == servers ? 0 : servers.size();

		if(size == 0) {
			return null;
		} else if(size == 1) {
			return servers.get(0);
		} else {
			// 根据hashcode计算位置
			int hashCode = null == cacheIdentifier ? 0 : cacheIdentifier.hashCode();
			int index = Math.abs(hashCode % size);
			return servers.get(index);
		}
	}
}

