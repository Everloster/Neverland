/**
 * ICache2.java
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

import com.nearme.base.cache.selector.CacheSelector;

/**
 * ClassName:ICache2
 * Function: 多级缓存接口
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-19  上午09:46:36
 */
public interface ICache2 {
	/**
	 * 获取
	 * @param cacheSelector
	 * @return 获取到的缓存数据
	 */
	<T> T get(CacheSelector cacheSelector) throws Exception;
	
	/**
	 * 更新
	 * @param cacheSelector
	 * @return 更新影响行数
	 */
	int update(CacheSelector cacheSelector) throws Exception;
	
	/**
	 * 添加
	 * @param cacheSelector
	 * @return 添加影响行数
	 */
	int insert(CacheSelector cacheSelector) throws Exception;
	
	/**
	 * 删除
	 * @param cacheSelector
	 * @return 删除影响行数
	 */
	int delete(CacheSelector cacheSelector) throws Exception;
	
	/**
	 * 资源释放
	 * @return
	 */
	void release();
	
	/**
	 * 是否支持数据自动过期
	 * @param 
	 * @return
	 */
	boolean expireSupported();
	
	/**
	 * 获取下一层缓存
	 * @return
	 */
	ICache2 getNextCache();

	/**
	 * 设置下一层缓存
	 * 如果当前层不支持数据自动过期，则无法设置下层缓存
	 * @param nextCache
	 */
	void setNextCache(ICache2 nextCache);
	
	/**
	 * 获取本层是否启用
	 * @return
	 */
	boolean isTierClosed();
	
	/**
	 * 设置本层是否启用
	 * @param isTierClosed
	 */
	void setTierClosed(boolean isTierClosed);
}

