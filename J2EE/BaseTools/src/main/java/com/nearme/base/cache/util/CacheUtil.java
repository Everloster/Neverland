/**
 * CacheUtil.java
 * com.nearme.base.cache.util
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-30 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nearme.base.cache.ICache2;
import com.nearme.base.cache.selector.CacheSelector;

/**
 * ClassName:CacheUtil <br>
 * Function: 缓存处理辅助类 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-1-30  下午06:59:35
 */
public final class CacheUtil {
	private static final Logger logger = LoggerFactory.getLogger(CacheUtil.class);

	/**
	 * 从缓存中获取数据如果未获取到,则重新计算并将结果存入缓存
	 * @param cache 缓存对象
	 * @param cacheSelector 缓存查询对象
	 * @param lastLoadTime 一级缓存数据最后加载时间，如果二级缓存的最后一次加载时间比一级缓存访问时间晚，<br>
	 * 			则一级缓存直接重新读取，如果不需要判断二级缓存的数据变更，可传入<value>Long.MAX_VALUE</value>
	 * @param rc 结果计算对象
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static <T> T getResult(ICache2 cache, CacheSelector cacheSelector, long lastLoadTime, IResultCompute<T> rc) {
		T t = null;

		//如果二级缓存的最后一次加载时间比一级缓存访问时间晚，则一级缓存直接重新读取
		if(lastLoadTime >= rc.getLastLoadTime()) {
			try {
				//尝试从缓存中获取
				t = (T)cache.get(cacheSelector);
			} catch (Exception e) {
				logger.warn("缓存获取失败，key:" + cacheSelector.getCacheKey(), e);
				//e.printStackTrace();
			}
		}

		//如果在缓存中获取到则直接返回
		if(null != t) {
			return t;
		}

		//重新计算结果
		t = rc.compute();
		if(null != t && !cache.isTierClosed()) {
			//得到的数据放入缓存
			try {
				cache.insert(new CacheSelector(cacheSelector.getCacheKey(), null, t, cacheSelector.getTimeout()));
			} catch (Exception e) {
				logger.warn("缓存插入失败，key:" + cacheSelector.getCacheKey(), e);
				//e.printStackTrace();
			}
		}

		return t;
	}
}

