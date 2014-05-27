/**
 * IGroupKeyHandler.java
 * com.nearme.base.cache.group
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-25 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache.group;

import java.util.List;

import com.nearme.base.cache.selector.CacheSelector;

/**
 * ClassName:IGroupKeyHandler
 * Function: 一组cache的key处理
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-25  下午03:37:05
 */
public interface IGroupKeyHandler<T> {
	/**
	 * 将多个CacheSelector合并为一个，以方便进行批量查询
	 * Function Description here
	 * @param 
	 * @return
	 */
	CacheSelector combin(List<CacheSelector> csList);
	
	/**
	 * 根据从存储中查到的值获取对应的CacheSelector，主要是cacheKey和timeout参数
	 * @param tObj 从存储中查到的值
	 * @return
	 */
	CacheSelector getRelateKey(T tObj);
}

