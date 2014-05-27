/**
 * Ehcache2.java
 * com.nearme.base.cache
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-2-13 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache;

import java.util.concurrent.atomic.AtomicBoolean;

import net.sf.ehcache.Cache;
import net.sf.ehcache.CacheManager;
import net.sf.ehcache.Element;

import com.nearme.base.cache.selector.CacheSelector;

/**
 * ClassName:Ehcache2 <br>
 * Function: ehcache实现的缓存 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-2-13  下午07:48:14
 */
public class Ehcache2 extends AbstractCache2 {

	private static AtomicBoolean cacheCreated = new AtomicBoolean(false);
	private static CacheManager cm;

	private String cacheName;	//缓存名

	@Override
	protected Object getCache(CacheSelector cacheSelector) throws Exception {
		Cache cache = getCache();
		Element ele = cache.get(cacheSelector.getCacheKey());

		return (null == ele) ? null : ele.getObjectValue();
	}

	@Override
	protected boolean setCache(CacheSelector cacheSelector, Object cacheValue)
			throws Exception {
		Cache cache = getCache();

		Element ele = new Element(cacheSelector.getCacheKey(), cacheValue);
		int timeout = (int)(cacheSelector.getTimeout() / 1000);
		ele.setTimeToLive(timeout);

		cache.put(ele);
		return true;
	}

	@Override
	protected int insertCache(CacheSelector cacheSelector) throws Exception {
		//如果只有这一层则在该层中插入
		if(null == this.getNextCache()) {
			return this.setCache(cacheSelector, cacheSelector.getSelectorValue()) ? 1 : 0;
		} else {
			return 0;
		}
	}

	@Override
	protected int updateCache(CacheSelector cacheSelector) throws Exception {
		//如果只有这一层则在该层中插入
		if(null == this.getNextCache()) {
			return this.setCache(cacheSelector, cacheSelector.getSelectorValue()) ? 1 : 0;
		} else {
			return 0;
		}
	}

	@Override
	protected int deleteCache(CacheSelector cacheSelector) throws Exception {
		Cache cache = getCache();

		return cache.remove(cacheSelector.getCacheKey()) ? 1 : 0;
	}

	/**
	 * 清空所有缓存
	 * @param
	 * @return
	 */
	public boolean flushAll() {

		getCache().removeAll();
		return true;
	}

	/**
	 * 获取ehcache的cache
	 * @param
	 * @return
	 */
	public Cache getCache() {
		initCacheManager();
		Cache cache = cm.getCache(cacheName);
		if(null == cache) {
			cm.addCacheIfAbsent(cacheName);

			return cm.getCache(cacheName);
		} else {
			return cache;
		}
	}

	public static void initCacheManager() {
		if(null == cm && cacheCreated.compareAndSet(false, true)) {
			cm = CacheManager.create();
		}
	}

	@Override
	public boolean expireSupported() {
		return true;
	}

	@Override
	public void releaseResource() {
	}

	/**
	 * 获取缓存对应组名称，方便统一管理
	 * @return  the cacheName
	 * @since   Ver 1.0
	 */
	public String getCacheName() {
		return cacheName;
	}

	/**
	 * 设置缓存对应组名称，方便统一管理
	 * @param   cacheName
	 * @since   Ver 1.0
	 */
	public void setCacheName(String cacheName) {
		this.cacheName = cacheName;
	}
}

