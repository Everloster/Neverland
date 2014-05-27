/**
 * RelateMemcache.java
 * com.oppo.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-13 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache;

import java.util.Map;

/**
 * ClassName:RelateMemcache
 * Function: 当指定的缓存key与其他缓存相关联(且关联关系确定)时，对关联的缓存进行处理
 * 如果需要设置其下层缓存，需要保证下层的cache层也实现了relateMap的处理
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-8-13  上午09:00:51
 */
public class RelateMemcache extends Memcache {
	public static final String KEY_SPLIT = ",";
	
	private Map<Memcache, String> relateMap;

	/**
	 * 获取key对应需要处理的key集合
	 * @return  the relateMap
	 * @since   Ver 1.0
	 */
	public Map<Memcache, String> getRelateMap() {
		return relateMap;
	}

	/**
	 * 设置key对应需要处理的key集合
	 * @param   relateMap    
	 * @since   Ver 1.0
	 */
	public void setRelateMap(Map<Memcache, String> relateMap) {
		this.relateMap = relateMap;
	}

	@Override
	public boolean setCache(String cacheId, Object obj, long timeout) {
		boolean isSucc = super.setCache(cacheId, obj, timeout);
		//如果添加成功，并且存在关联key则删除相应的关联key
		if(isSucc) {
			clearRelateMap();
		}
		
		return isSucc;
	}
	
	@Override
	public int deleteCache(String cacheId, StorageObject storageObject, long timeout) {
		int deleteCount = super.deleteCache(cacheId, storageObject, timeout);
		//如果删除成功，并且存在关联key则更新相应的关联key
		if(deleteCount > 0) {
			clearRelateMap();
		}
		
		return deleteCount;
	}
	
	/**
	 * 清空对应Map中的key
	 * @param 
	 * @return
	 */
	protected void clearRelateMap() {
		if(null != relateMap && relateMap.size() > 0) {
			for(Memcache mem : relateMap.keySet()) {
				String[] keys = relateMap.get(mem).split(KEY_SPLIT);
				for(String key : keys) {
					mem.deleteCache(key, null, 0);
				}
			}
		}
	}
	
	/**
	 * Function Description here
	 * @param 
	 * @return
	 */

	public static void main(String[] args) {

		// TODO Auto-generated method stub

	}

}

