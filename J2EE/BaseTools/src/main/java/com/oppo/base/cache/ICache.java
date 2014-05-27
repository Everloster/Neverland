/**
 * ICache.java
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

/**
 * ClassName:ICache
 * Function: 缓存接口
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-8  上午10:57:48
 */
public interface ICache {
	/**
	 * 获取
	 * @param cacheId
	 * @param sqlObject
	 * @param timeout 相对于当前时间的过期时间,以毫秒为单位
	 * @return
	 */
	Object get(String cacheId, StorageObject storageObject, long timeout) throws Exception;
	
	/**
	 * 更新
	 * @param cacheId
	 * @param sql
	 * @return
	 */
	int update(String cacheId, StorageObject storageObject, long timeout) throws Exception;
	
	/**
	 * 添加
	 * @param cacheId
	 * @param sqlObject
	 * @param timeout 相对于当前时间的过期时间,以毫秒为单位
	 * @return
	 */
	int insert(String cacheId, StorageObject storageObject, long timeout) throws Exception;
	
	/**
	 * 删除
	 * @param cacheId
	 * @param sqlObject
	 * @param timeout 相对于当前时间的过期时间,以毫秒为单位
	 * @return
	 */
	int delete(String cacheId, StorageObject storageObject, long timeout) throws Exception;

	/**
	 * 获取下一层缓存
	 * @return
	 */
	ICache getNextCache();

	/**
	 * 设置下一层缓存
	 * @param nextCache
	 */
	void setNextCache(ICache nextCache);
	
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
	
	/**
	 * 资源释放
	 * @param 
	 * @return
	 */
	void release();
}

