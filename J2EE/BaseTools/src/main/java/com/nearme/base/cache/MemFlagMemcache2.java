/**
 * MemFlagMemcache2.java
 * com.nearme.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-12-14 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache;

import com.danga.MemCached.MemCachedClient;

/**
 * ClassName:MemFlagMemcache2 <br>
 * Function: key带标记的memcache,标记将保存到memcache中 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-12-14  上午09:12:14
 */
public class MemFlagMemcache2 extends Memcache2 {
	/**
	 * 将传入的cacheId转换为最终存储的cacheId,在转换后增加一个标记，
	 * @param cacheId
	 * @return
	 */
	@Override
	protected String transferCacheId(String cacheId) {
		String key = super.transferCacheId(cacheId);
		
		return key + getKeyFlag();
	}
	
	/**
	 * 获取缓存key的标记,此处为利用文件进行存储 <br>
	 * 如果需要改为其他方式存储flag,则需要重写此方法 <br>
	 * @param 
	 * @return
	 */
	protected String getKeyFlag() {
		MemCachedClient memCachedClient = this.getMemCachedClient();
		
		//flag对应的cache key
		String flagKey = getCacheFlag();
		
		return String.valueOf(memCachedClient.get(flagKey));
	}
	
	/**
	 * 升级缓存key的版本
	 * @return 更新后的版本
	 */
	public String updateKeyVersion() {
		String flagKey = getCacheFlag();
		MemCachedClient memCachedClient = this.getMemCachedClient();
		
		return String.valueOf(memCachedClient.addOrIncr(flagKey, 1));
	}
	
	private String getCacheFlag() {
		return "CacheFlag_" + this.getPrefix();
	}
}

