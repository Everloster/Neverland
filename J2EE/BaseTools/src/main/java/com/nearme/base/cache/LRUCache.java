/**
 * LRUCache.java
 * com.nearme.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-27 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache;

import java.util.concurrent.ConcurrentHashMap;

import com.nearme.base.cache.local.CacheObject;
import com.nearme.base.cache.local.LocalCacheMap;
import com.nearme.base.cache.local.LocalCacheMapMonitor;
import com.nearme.base.cache.selector.CacheSelector;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:LRUCache <br>
 * Function: LRU算法管理的缓存 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-27  下午03:42:20
 */
public class LRUCache extends AbstractCache2 {
	//保存缓存map的map
	private static final ConcurrentHashMap<String, LocalCacheMap<String, CacheObject>> poolMap 
		= new ConcurrentHashMap<String, LocalCacheMap<String, CacheObject>>(128);

	//默认的池名为default
	public static final String DEFAULT_POOL_NAME = "default";
	//默认的最小内存保持为32M
	public static final long DEFAULT_MIN_FREE_MEMORY = 32 * 1024 * 1024l;
	
	//cache队列最大长度
	public static final int DEFAULT_MAX_CACHE_LENGTH = 131072;
	
	private String poolName;

	public LRUCache() {
		this(DEFAULT_POOL_NAME, true, DEFAULT_MIN_FREE_MEMORY);
	}
	
	/**
	 * 初始化一个LRUCache的实例
	 *
	 * @param poolName 缓存用到的池名，可传空，默认为default
	 * @param autoExpire 是否自动清理缓存中的过期数据
	 * @param minFreeMemory 最大预留的内存空间大小
	 */
	public LRUCache(String poolName, boolean autoExpire, long minFreeMemory) {
		super();
	
		init(poolName, autoExpire, minFreeMemory);
	}
	
	/**
	 * 初始化缓存池
	 * @param 
	 * @return
	 */
	public synchronized void init(String poolName, boolean autoExpire, long minFreeMemory) {
		if(StringUtil.isNullOrEmpty(poolName)) {
			//设置默认pool name
			poolName = DEFAULT_POOL_NAME;
		}
		
		this.poolName = poolName;
		//如果不存在则初始化
		if(!poolMap.containsKey(poolName)) {
			poolMap.putIfAbsent(poolName, new LocalCacheMap<String, CacheObject>(DEFAULT_MAX_CACHE_LENGTH));
		}
		
		//设置自定过期，则会定期对缓存进行清理
		if(autoExpire) {
			LocalCacheMapMonitor.getInstance(minFreeMemory).addCacheMap(poolName, getCacheMap());
		}
	}
	
	/**
	 * 获取当前实例对应的poolName对应的缓存
	 * @param 
	 * @return
	 */
	public LocalCacheMap<String, CacheObject> getCacheMap() {
		return getCacheMap(poolName);
	}
	
	/**
	 * 获取对应poolName对应的缓存
	 * @param 
	 * @return
	 */
	public LocalCacheMap<String, CacheObject> getCacheMap(String poolName) {
		if(StringUtil.isNullOrEmpty(poolName)) {
			//设置默认pool name
			poolName = DEFAULT_POOL_NAME;
		}
		
		return poolMap.get(poolName);
	}
	
	/**
	 * 清空当前缓存对应池的缓存
	 * @param 
	 * @return
	 */
	public void flush() {
		flush(poolName);
	}
	
	/**
	 * 清空指定池的缓存
	 * @param 
	 * @return
	 */
	public void flush(String poolName) {
		LocalCacheMap<String, CacheObject> cacheMap = getCacheMap(poolName);
		if(null != cacheMap) {
			cacheMap.clear();
		}
	}
	
	/**
	 * 清空所有缓存
	 * @param 
	 * @return
	 */
	public void flushAll() {
		for(LocalCacheMap<String, CacheObject> cacheMap : poolMap.values()) {
			cacheMap.clear();
		}
	}
	
	@Override
	public boolean expireSupported() {
		return true;
	}

	@Override
	public Object getCache(CacheSelector cacheSelector) throws Exception {
		LocalCacheMap<String, CacheObject> cacheMap = getCacheMap();
		String key = cacheSelector.getCacheKey();
		CacheObject co = cacheMap.get(key);
		if(null == co) {
			return null;
		} else {
			//如果已经过期则直接移除
			if(co.isCacheExpire()) {
				cacheMap.remove(key);
				return null;
			} else {
				return co.getValue();
			}
		}
	}

	@Override
	public boolean setCache(CacheSelector cacheSelector, Object cacheValue)
			throws Exception {
		LocalCacheMap<String, CacheObject> cacheMap = getCacheMap();
		
		String key = cacheSelector.getCacheKey();
		CacheObject co = new CacheObject(key, cacheValue, cacheSelector.getTimeout());
		cacheMap.put(key, co);
		
		return true;
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

	@Override
	public int deleteCache(CacheSelector cacheSelector) throws Exception {
		LocalCacheMap<String, CacheObject> cacheMap = getCacheMap();
		
		String key = cacheSelector.getCacheKey();
		long timeout = cacheSelector.getTimeout();
		if(timeout > 0) {
			//延迟删除
			CacheObject co = cacheMap.get(key);
			if(null == co) {
				return 0;
			} else if(co.isCacheExpire()) {
				//已过期的不处理
				cacheMap.remove(key);
				return 0;
			} else {
				//设置指定时间后无效
				co.setExpireTime(timeout);
				cacheMap.put(key, co);
				return 1;
			}
		} else {
			return (null == cacheMap.remove(key)) ? 0 : 1;
		}
	}

	@Override
	public void releaseResource() {
	}

	/**
	 * 获取poolName
	 * @return  the poolName
	 * @since   Ver 1.0
	 */
	public String getPoolName() {
		return poolName;
	}

	/**
	 * 设置poolName
	 * @param   poolName    
	 * @since   Ver 1.0
	 */
	public void setPoolName(String poolName) {
		this.poolName = poolName;
	}
	
	public static void main(String[] args) throws Exception {
		ICache2 cache = new LRUCache();
		
		//直接获取
		Object obj = cache.get(new CacheSelector("key"));
		System.out.println(obj);
		
		//插入后获取
		cache.insert(new CacheSelector("key", null, new int[]{1,2,3}, 1));
		obj = cache.get(new CacheSelector("key"));
		System.out.println(obj);
	
		//删除后获取
		cache.delete(new CacheSelector("key", null, "value1", 0));
		obj = cache.get(new CacheSelector("key"));
		System.out.println(obj);
	}
}

