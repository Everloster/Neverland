/**
 * MultiMemcache.java
 * com.oppo.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-26 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache;

import com.oppo.base.cache.util.MultiMemcacheMonitor;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:MultiMemcache
 * Function: 当指定的缓存key与其他缓存相关联(且关联关系不确定)时，对关联的缓存进行处理
 * 如果需要设置其下层缓存，需要保证下层的cache层也实现了multiKey的处理
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-26  上午09:11:33
 */
public class MultiMemcache extends Memcache {
	private String multiKey;

	/**
	 * 获取集合key
	 * @return  the multiKey
	 * @since   Ver 1.0
	 */
	public String getMultiKey() {
		return multiKey;
	}

	/**
	 * 设置集合key
	 * @param   multiKey    
	 * @since   Ver 1.0
	 */
	public void setMultiKey(String multiKey) {
		this.multiKey = multiKey;
	}
	
	@Override
	public boolean setCache(String cacheId, Object obj, long timeout) {
		boolean isSucc = super.setCache(cacheId, obj, timeout);
		//如果添加成功，并且存在关联key则更新相应的关联key
		if(isSucc && !StringUtil.isNullOrEmpty(multiKey)) {
			MultiMemcacheMonitor.getInstance().updateMultiKey(multiKey, cacheId, timeout, this);
		}
		
		return isSucc;
	}
	
	@Override
	public int deleteCache(String cacheId, StorageObject storageObject, long timeout) {
		int deleteCount = super.deleteCache(cacheId, storageObject, timeout);
		//如果删除成功，并且存在关联key则更新相应的关联key
		if(deleteCount > 0 && !StringUtil.isNullOrEmpty(multiKey)) {
			MultiMemcacheMonitor.getInstance().deleteMultiKey(multiKey, cacheId, this);
		}
		
		return deleteCount;
	}
}

