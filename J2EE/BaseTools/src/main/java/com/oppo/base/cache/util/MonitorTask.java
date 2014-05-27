/**
 * MonitorTask.java
 * com.oppo.base.cache.util
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-11 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache.util;

import com.oppo.base.cache.Memcache;

/**
 * ClassName:MonitorTask
 * Function: 监控任务实体
 * @author   80036381
 * @version  MultiMemcacheMonitor
 * @since    Ver 1.1
 * @Date	 2011-7-26  上午09:22:46
 * @see
 */
class MonitorTask {
	public static final int DELETE = 0;
	public static final int UPDATE = 0;
	
	private String multiKey;
	private String memcacheKey;
	private long expire;
	private Memcache memcache;
	private int type;
	
	public MonitorTask() {
	}
	
	public MonitorTask(String multiKey, String memcacheKey, long expire, Memcache memcache, int type) {
		this.multiKey = multiKey;
		this.memcacheKey = memcacheKey;
		if(expire == 0) {
			this.expire = Long.MAX_VALUE;
		} else {
			this.expire = expire;
		}
		this.memcache = memcache;
		this.type = type;
	}
	
	/**
	 * 获取关联key
	 * @return  the multiKey
	 * @since   Ver 1.0
	 */
	public String getMultiKey() {
		return multiKey;
	}
	
	/**
	 * 设置关联key
	 * @param   multiKey    
	 * @since   Ver 1.0
	 */
	public void setMultiKey(String multiKey) {
		this.multiKey = multiKey;
	}
	
	/**
	 * 获取缓存key
	 * @return  the memcacheKey
	 * @since   Ver 1.0
	 */
	public String getMemcacheKey() {
		return memcacheKey;
	}
	
	/**
	 * 设置缓存key
	 * @param   memcacheKey    
	 * @since   Ver 1.0
	 */
	public void setMemcacheKey(String memcacheKey) {
		this.memcacheKey = memcacheKey;
	}
	
	/**
	 * 获取过期时间,毫秒为单位
	 * @return  the expire
	 * @since   Ver 1.0
	 */
	public long getExpire() {
		return expire;
	}
	
	/**
	 * 设置过期时间,毫秒为单位
	 * @param   expire    
	 * @since   Ver 1.0
	 */
	public void setExpire(long expire) {
		this.expire = expire;
	}
	
	/**
	 * 获取对应的memcache
	 * @return  the memcache
	 * @since   Ver 1.0
	 */
	public Memcache getMemcache() {
		return memcache;
	}
	
	/**
	 * 设置对应的memcache
	 * @param   memcache    
	 * @since   Ver 1.0
	 */
	public void setMemcache(Memcache memcache) {
		this.memcache = memcache;
	}
	
	/**
	 * 获取操作类型
	 * @return  the type
	 * @since   Ver 1.0
	 */
	public int getType() {
		return type;
	}
	
	/**
	 * 设置操作类型
	 * @param   type    
	 * @since   Ver 1.0
	 */
	public void setType(int type) {
		this.type = type;
	}
}

