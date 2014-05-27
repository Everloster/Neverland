/**
 * IGroupHandler.java
 * com.oppo.base.cache.group
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-25 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.cache.group;

import java.util.List;

import com.oppo.base.cache.StorageObject;

/**
 * ClassName:IGroupHandler
 * Function: 一组cache的key处理
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-25  下午05:22:42
 */
public interface IGroupHandler<T> {
	/**
	 * 将多个CacheSelector合并为一个，以方便进行批量查询
	 * Function Description here
	 * @param 
	 * @return
	 */
	StorageObject combin(List<StorageObject> csList);
	
	/**
	 * 根据从存储中查到的值获取对应的cacheKey
	 * @param tObj 从存储中查到的值
	 * @return
	 */
	String getRelateKey(T tObj);
}

