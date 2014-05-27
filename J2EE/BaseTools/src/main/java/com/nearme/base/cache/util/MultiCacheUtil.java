/**
 * MultiCacheUtil.java
 * com.nearme.base.cache
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-21 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache.util;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;

import com.nearme.base.cache.Memcache2;
import com.nearme.base.cache.selector.CacheSelector;
import com.nearme.base.concurrent.ExecutorManager;


/**
 * ClassName:MultiCacheUtil
 * Function: 一次需要用多个key进行查询时
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-10-21  下午03:33:58
 */
public class MultiCacheUtil {
	private static final ExecutorService EXECUTOR_SERVICE = ExecutorManager.newCachedThreadPool();

	/**
	 * 当一个memcache key与其他不确定的key关联,则使用此方法异步的把这些key关联起来
	 * @param cacheSelector
	 * @param groupName		关联一组key的组名
	 * @param mem
	 * @see com.nearme.base.cache.util.MultiCacheUtil#updateGroupMemcache(String groupName, Memcache2 mem)
	 * @return
	 */
	public static void updateGroupMemcacheAsync(final CacheSelector cacheSelector, final String groupName, final Memcache2 mem) {
		EXECUTOR_SERVICE.execute(new Runnable(){
			@Override
			public void run() {
				try {
					updateGroupMemcache(cacheSelector, groupName, mem);
				} catch(Exception ex) {

				}
			}
		});
	}

	/**
	 * 当一个memcache key与其他不确定的key关联,则使用此方法把这些key关联起来
	 * @param cacheSelector
	 * @param groupName		关联一组key的组名
	 * @param mem
	 * @see com.nearme.base.cache.util.MultiCacheUtil#updateGroupMemcache(String groupName, Memcache2 mem)
	 * @return
	 */
	public static void updateGroupMemcache(CacheSelector cacheSelector, String groupName, Memcache2 mem)
	throws Exception {
		CacheSelector groupCacheSelector = new CacheSelector(groupName);
		Map<String, Long> oldGroup = mem.get(groupCacheSelector);

		long now = System.currentTimeMillis();

		String cacheKey = cacheSelector.getCacheKey();
		long timeout = cacheSelector.getTimeout();
		if(timeout == 0) {
			timeout = Long.MAX_VALUE;
		} else {
			timeout = now + timeout;
		}

		int size = (null == oldGroup) ? 0 : oldGroup.size();
		Map<String, Long> newGroup = new HashMap<String, Long>(size + 1);
		boolean groupChange = false;
		boolean containNewKey = false;
		if(size > 0) {
			//去掉过期的
			for(String key : oldGroup.keySet()) {
				long to = oldGroup.get(key);
				if(to > now) {
					newGroup.put(key, to);
					//如果和要插入的相等则不用新加
					if(key.equals(cacheKey)) {
						containNewKey = true;
					}
				} else {
					groupChange = true;
				}
			}
		}

		if(!containNewKey) {
			newGroup.put(cacheKey, timeout);
			groupChange = true;
		}

		if(groupChange) {
			//将group key加入到memcache中
			mem.setCache(groupCacheSelector, newGroup);
		}
	}

	/**
	 * 异步删除一组相关联的key时使用
	 * @param groupName		关联一组key的组名
	 * @param mem
	 * @return
	 */
	public static void updateGroupMemcacheAsync(final String groupName, final Memcache2 mem)
	throws Exception {
		EXECUTOR_SERVICE.execute(new Runnable(){
			@Override
			public void run() {
				try {
					deleteGroupMemcache(groupName, mem);
				} catch(Exception ex) {

				}
			}
		});
	}

	/**
	 * 删除一组相关联的key时使用
	 * @param groupName		关联一组key的组名
	 * @param mem
	 * @return
	 */
	public static void deleteGroupMemcache(String groupName, Memcache2 mem)
	throws Exception {
		CacheSelector groupCacheSelector = new CacheSelector(groupName);
		Map<String, Long> oldGroup = mem.get(groupCacheSelector);
		if(null != oldGroup) {
			//删除所有缓存
			for(String key : oldGroup.keySet()) {
				mem.delete(new CacheSelector(key));
			}

			//删除group key
			mem.delete(groupCacheSelector);
		}
	}
}

