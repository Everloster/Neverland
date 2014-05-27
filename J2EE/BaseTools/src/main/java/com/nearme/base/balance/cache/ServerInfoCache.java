/**
 * ServerInfoCache.java
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

import com.nearme.base.balance.model.ServerInfo;

/**
 * ClassName:ServerInfoCache <br>
 * Function: 缓存的服务信息 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-6-17  上午9:52:59
 */
public class ServerInfoCache {
	private long cacheStartTime;
	private ServerInfo server;

	public ServerInfoCache(ServerInfo server) {
		this.server = server;
		this.cacheStartTime = System.currentTimeMillis();
	}

	/**
	 * 获取cacheStartTime
	 * @return  the cacheStartTime
	 * @since   Ver 1.0
	 */
	public long getCacheStartTime() {
		return cacheStartTime;
	}
//	/**
//	 * 设置cacheStartTime
//	 * @param   cacheStartTime
//	 * @since   Ver 1.0
//	 */
//	public void setCacheStartTime(long cacheStartTime) {
//		this.cacheStartTime = cacheStartTime;
//	}
	/**
	 * 获取server
	 * @return  the server
	 * @since   Ver 1.0
	 */
	public ServerInfo getServer() {
		return server;
	}
//	/**
//	 * 设置server
//	 * @param   server
//	 * @since   Ver 1.0
//	 */
//	public void setServer(ServerInfo server) {
//		this.server = server;
//	}
}

