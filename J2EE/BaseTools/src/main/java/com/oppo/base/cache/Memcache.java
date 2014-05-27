/**
 * Memcache.java
 * com.oppo.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-6-8 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache;

import java.util.Date;

import com.danga.MemCached.MemCachedClient;
import com.oppo.base.security.MD5;

/**
 * ClassName:Memcache
 * Function: Memcache缓存实现
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-8  上午11:20:25
 */
public class Memcache extends AbstractCache {
	
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
	
	public Memcache() {
		super();
	}

	/**
	 * 初始化MemCachedClient,所有操作应该调用此方法
	 * @param 
	 * @return
	 */
	protected MemCachedClient getMemCachedClient() {
		if(null == memCache){
			memCache = new MemCachedClient(poolName);
		}

		return memCache;
	}
	
	@Override
	public int deleteCache(String cacheId, StorageObject storageObject, long timeout) {

		cacheId = transferCacheId(cacheId);
		if(timeout <= 0){
			return getMemCachedClient().delete(cacheId) ? 1 : 0;
		} else {
			return getMemCachedClient().delete(cacheId, getDate(timeout)) ? 1 : 0;
		}
	}

	@Override
	public Object getCache(String cacheId, StorageObject storageObject) {
		cacheId = transferCacheId(cacheId);
		
		return getMemCachedClient().get(cacheId);
	}
	
	@Override
	public boolean setCache(String cacheId, Object obj, long timeout) {
		cacheId = transferCacheId(cacheId);
		boolean isSucc;
		if(timeout <= 0) {
			isSucc = getMemCachedClient().set(cacheId, obj);
		} else {
			isSucc = getMemCachedClient().set(cacheId, obj, getDate(timeout));
		}

		return isSucc;
	}

	@Override
	public int insertCache(String cacheId, StorageObject storageObject, long timeout) throws Exception {
		//如果只有这一层则在该层中插入
		if(null == this.getNextCache()) {
			return this.setCache(cacheId, storageObject.getStorageObject(), timeout) ? 1 : 0;
		} else {
			return 0;
		}
	}

	@Override
	public int updateCache(String cacheId, StorageObject storageObject, long timeout) throws Exception {
		//如果只有这一层则在该层中更新
		if(null == this.getNextCache()) {
			return this.setCache(cacheId, storageObject.getStorageObject(), timeout) ? 1 : 0;
		} else {
			return 0;
		}
	}

	/**
	 * 获取多个key的值,key会进行转换
	 * @param 
	 * @return
	 */
	public Object[] getCache(String[] cacheIds) {
		int len = cacheIds.length;
		//组合出key数组
		String[] finalKeys =  new String[len];
		for(int i = 0; i < len; i++) {
			finalKeys[i] = transferCacheId(cacheIds[i]);
		}
		
		return this.getCacheWithoutTransfer(finalKeys);
	}
	
	/**
	 * 获取多个key的值,key不会再进行转换
	 * @param 
	 * @return
	 */
	public Object[] getCacheWithoutTransfer(String[] cacheIds) {
		return this.getMemCachedClient().getMultiArray(cacheIds);
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

	@Override
	public void releaseResource() {
	}
}

