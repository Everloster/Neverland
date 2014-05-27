/**
 * DBConnectForList.java
 * com.oppo.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-11 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache;
/**
 * ClassName:DBConnectForList
 * Function: 可查询list的DB连接
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-11  上午08:48:22
 */
public class DBConnectForList extends DBConnect {
	@Override
	public Object getCache(String cacheId, StorageObject storageObject) throws Exception {
		return getSqlSession().selectList(storageObject.getStorageId(), storageObject.getStorageObject());
	}
}

