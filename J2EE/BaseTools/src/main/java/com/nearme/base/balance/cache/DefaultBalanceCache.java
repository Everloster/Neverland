/**
 * DefaultBalanceCache.java
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
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import com.nearme.base.balance.model.ServerInfo;
import com.oppo.base.common.OConstants;

/**
 * ClassName:DefaultBalanceCache <br>
 * Function: 默认实现，将标记的后指定位数值作为真实的标记 <br>
 * 如 identifierLen = 3, cacheIdentifier = "1314", 则真实使用的缓存标记为  "314"<br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-28  下午1:57:15
 */
public class DefaultBalanceCache implements IBalanceCache {

	private ConcurrentMap<String, ConcurrentMap<String, ServerInfoCache>> cache = 
			new ConcurrentHashMap<String, ConcurrentMap<String, ServerInfoCache>>();
	private int identifierLen;
	private long cacheTimeout;

	private Random rand = new Random();

	public DefaultBalanceCache() {
		// 默认截取最后三位
		setIdentifierLen(3);

		// 默认缓存30分钟
		setCacheTimeout(30 * 60 * 1000);
	}

	@Override
	public ServerInfo getCacheServer(String groupId, String cacheIdentifier) {
		String realIdentifier = getRealIdentifier(cacheIdentifier);

		ConcurrentMap<String, ServerInfoCache> group = getCacheByGroup(groupId);
		ServerInfoCache ci = group.get(realIdentifier);
		if (null != ci) {
			if (cacheTimeout < 1) {
				// 小于1则缓存不过期
				return ci.getServer();
			} else if (System.currentTimeMillis() - ci.getCacheStartTime() < cacheTimeout) {
				// 缓存信息时间未超过过期时间则直接返回
				return ci.getServer();
			} else {
				cache.remove(realIdentifier);
			}
		}

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
			//按照权重获取server
			int totalWeight = servers.get(0).getServerGroup().getTotalWeight();
			int weight = rand.nextInt(totalWeight);
			int tmpWeight = 0;
			for(int i = 0; i < size; i++) {
				ServerInfo server = servers.get(i);
				tmpWeight += server.getWeight();
				if(weight < tmpWeight) {
					return server;
				}
			}

			//随机获取server
			return servers.get(rand.nextInt(size - 1));
		}
	}

	@Override
	public void putServerToCache(String groupId, String cacheIdentifier, ServerInfo serverInfo) {
		String realIdentifier = getRealIdentifier(cacheIdentifier);

		ConcurrentMap<String, ServerInfoCache> group = getCacheByGroup(groupId);
		if (null == serverInfo) {
			group.remove(realIdentifier);
		} else {
			ServerInfoCache ci = new ServerInfoCache(serverInfo);
			group.putIfAbsent(realIdentifier, ci);
		}
	}
	
	protected ConcurrentMap<String, ServerInfoCache> getCacheByGroup(String groupId) {
		ConcurrentMap<String, ServerInfoCache> group = cache.get(groupId);
		if (null == group) {
			group = new ConcurrentHashMap<String, ServerInfoCache>();
			ConcurrentMap<String, ServerInfoCache> oldGroup = cache.putIfAbsent(groupId, group);
			if (null != oldGroup) {
				group = oldGroup;
			}
		}
		
		return group;
	}

	protected String getRealIdentifier(String inputIdentifier) {
		if (null == inputIdentifier) {
			return OConstants.EMPTY_STRING;
		}

		int len = inputIdentifier.length();
		if (len > identifierLen) {
			return inputIdentifier.substring(len - identifierLen);
		} else {
			return inputIdentifier;
		}
	}

	/**
	 * 获取标记位被截取的长度
	 * @return  the identifierLen
	 * @since   Ver 1.0
	 */
	public int getIdentifierLen() {
		return identifierLen;
	}


	/**
	 * 设置标记位被截取的长度,不可小于1
	 * @param   identifierLen
	 * @since   Ver 1.0
	 */
	public void setIdentifierLen(int identifierLen) {
		if (identifierLen < 1) {
			throw new IllegalArgumentException("identifierLen must lager than 0, input is " + identifierLen);
		}

		this.identifierLen = identifierLen;
	}

	/**
	 * 获取缓存超时时间（毫秒）
	 * @return  the cacheTimeout
	 * @since   Ver 1.0
	 */
	public long getCacheTimeout() {
		return cacheTimeout;
	}

	/**
	 * 设置缓存超时时间（毫秒），小于1则不过期
	 * @param   cacheTimeout
	 * @since   Ver 1.0
	 */
	public void setCacheTimeout(long cacheTimeout) {
		this.cacheTimeout = cacheTimeout;
	}

	public static void main(String[] args) {
		DefaultBalanceCache b = new DefaultBalanceCache();
		System.out.println(b.getRealIdentifier("asd"));
		System.out.println(b.getRealIdentifier("1"));
		System.out.println(b.getRealIdentifier("asd1"));
		System.out.println(b.getRealIdentifier("asd11"));
		System.out.println(b.getRealIdentifier("asd112"));
	}
}

