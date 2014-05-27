/**
 * Memcache2.java
 * com.nearme.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-19 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache;

import java.util.Date;

import com.danga.MemCached.MemCachedClient;
import com.nearme.base.cache.selector.CacheSelector;
import com.oppo.base.security.MD5;

/**
 * ClassName:Memcache2
 * Function: memcache实现
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-19  上午11:21:07
 */
public class Memcache2 extends AbstractCache2 {
	/**
	 * 无更新模式
	 */
	public static final int UPDATE_TYPE_NORMAL = 0;
	
	/**
	 * memcache更新标记模式
	 */
	public static final int UPDATE_TYPE_MEMCACHE = 1;
	
	/**
	 * xml更新标记模式
	 */
	public static final int UPDATE_TYPE_XML = 2;
	
	private String poolName;
	private String prefix;
	private MemCachedClient memCache;
	
	/**
	 * 缓存池名字段
	 */
	public static final String CONFIG_POOL_NAME = "PoolName";
	
	/**
	 * 缓存服务器列表字段
	 */
	public static final String CONFIG_SERVER_LIST = "ServerList";
	
	/**
	 * 缓存权重字段，权重之间以半角逗号分隔
	 */
	public static final String CONFIG_SERVER_WEIGHT = "ServerWeight";
	
	/**
	 * 初始连接数字段
	 */
	public static final String CONFIG_INIT_CONNECTION = "InitConnection";
	
	/**
	 * 最小连接数字段
	 */
	public static final String CONFIG_MIN_CONNECTION = "MinConnection";
	
	/**
	 * 最大连接数字段
	 */
	public static final String CONFIG_MAX_CONNECTION = "MaxConnection";
	
	/**
	 * 最大空闲时间字段
	 */
	public static final String CONFIG_MAX_IDEL = "MaxIdelTime";
	
	public static final String CONFIG_MAX_BUSY = "MaxBusyTime";
	public static final String CONFIG_MAINT_SLEEP = "MaintSleep";
	public static final String CONFIG_READ_TIMEOUT = "SocketReadTimeout";
	public static final String CONFIG_CONNECT_TIMEOUT = "SocketConnectTimeout";
	public static final String CONFIG_NAGLE_ALGORITHM = "NagleAlgorithm";
	public static final String CONFIG_HASHING_ALGORITHM = "HashingAlgorithm";
	public static final String CONFIG_FAILOVER = "Failover";
	
	public Memcache2() {
		super();
	}

	/**
	 * 初始化MemCachedClient,所有操作应该调用此方法
	 * @param 
	 * @return
	 */
	public MemCachedClient getMemCachedClient() {
		if(null == memCache){
			memCache = new MemCachedClient(poolName);
		}
		
		return memCache;
	}
	
	@Override
	public int deleteCache(CacheSelector cacheSelector) {
		String cacheId = transferCacheId(cacheSelector.getCacheKey());
		
		long timeout = cacheSelector.getTimeout();
		if(timeout <= 0){
			return getMemCachedClient().delete(cacheId) ? 1 : 0;
		} else {
			return getMemCachedClient().delete(cacheId, getDate(timeout)) ? 1 : 0;
		}
	}

	@Override
	public Object getCache(CacheSelector cacheSelector) {
		String cacheId = transferCacheId(cacheSelector.getCacheKey());
		
		return getMemCachedClient().get(cacheId);
	}
	
	@Override
	public boolean setCache(CacheSelector cacheSelector, Object cacheValue) {
		String cacheId = transferCacheId(cacheSelector.getCacheKey());
		boolean isSucc;
		long timeout = cacheSelector.getTimeout();
		if(timeout <= 0) {
			isSucc = getMemCachedClient().set(cacheId, cacheValue);
		} else {
			isSucc = getMemCachedClient().set(cacheId, cacheValue, getDate(timeout));
		}

		return isSucc;
	}

	@Override
	public int insertCache(CacheSelector cacheSelector) throws Exception {
		//如果只有这一层则在该层中插入
		if(null == this.getNextCache()) {
			return this.setCache(cacheSelector, cacheSelector.getSelectorValue()) ? 1 : 0;
		} else {
			return 0;
		}
	}

	@Override
	public int updateCache(CacheSelector cacheSelector) throws Exception {
		//如果只有这一层则在该层中更新
		if(null == this.getNextCache()) {
			return this.setCache(cacheSelector, cacheSelector.getSelectorValue()) ? 1 : 0;
		} else {
			return 0;
		}
	}
	
	/**
	 * 获取一组key对应的缓存,key会进行转换
	 * @param 
	 * @return
	 */
	public Object[] getCache(String[] cacheKeys) {
		//首先对key进行转换
		int len = cacheKeys.length;
		String[] finalKeys = new String[len];
		for(int i = 0; i < len; i++) {
			finalKeys[i] = transferCacheId(cacheKeys[i]);
		}
		
		//获取缓存值
		return getCacheWithoutTransfer(finalKeys);
	}
	
	/**
	 * 获取一组key对应的缓存,key不会进行转换
	 * @param 
	 * @return
	 */
	public Object[] getCacheWithoutTransfer(String[] finalKeys) {
		if(this.isTierClosed()) {
			return new Object[finalKeys.length];
		}
		
		return getMemCachedClient().getMultiArray(finalKeys);
	}
	
	@Override
	public void releaseResource() {
	}
	
	@Override
	public boolean expireSupported() {
		return true;
	}
	
	/**
	 * 清空所有缓存
	 * @param 
	 * @return
	 */
	public boolean flushAll() {
		return getMemCachedClient().flushAll();
	}
	
	public String getPoolName() {
		return poolName;
	}

	/**
	 * 设置memcache使用的poolname
	 * @param poolName
	 */
	public void setPoolName(String poolName) {
		this.poolName = poolName;
	}
	
	public String getPrefix() {
		return prefix;
	}

	/**
	 * 设置cache前缀,为了区分各个id,各类缓存前缀应该设为不同的
	 * @param type
	 */
	public void setPrefix(String prefix) {
		this.prefix = prefix;
	}
	
	/**
	 * 将传入的cacheId转换为最终存储的cacheId
	 * @param 
	 * @return
	 */
	protected String transferCacheId(String cacheId) {
		return prefix + MD5.md5(cacheId.toLowerCase());
	}
	
	/**
	 * 根据过期时间获取最终需要的java.util.Date对象
	 * @param 
	 * @return
	 */
	protected Date getDate(long timeout) {
		return new Date(System.currentTimeMillis() + timeout);
	}
}

