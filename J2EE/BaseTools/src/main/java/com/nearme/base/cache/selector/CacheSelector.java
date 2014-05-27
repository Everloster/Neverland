/**
 * CacheSelector.java
 * com.nearme.base.cache.selector
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-19 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache.selector;

/**
 * ClassName:CacheSelector
 * Function: 缓存操作时的传值对象
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-19  上午09:47:02
 */
public class CacheSelector {
	private String cacheKey;
	private String selectorKey;
	private Object selectorValue;
	private long timeout;
	private Object emptyValue;
	
	public CacheSelector() {
	}
	
	/**
	 * Creates a new instance of CacheSelector.
	 *
	 * @param cacheKey		缓存key
	 */
	public CacheSelector(String cacheKey) {
		this(cacheKey, null, null, 0, null);
	}
	
	/**
	 * Creates a new instance of CacheSelector.
	 *
	 * @param cacheKey		缓存key
	 * @param timeout		缓存过期时间
	 */
	public CacheSelector(String cacheKey, long timeout) {
		this(cacheKey, null, null, timeout, null);
	}
	
	/**
	 * Creates a new instance of CacheSelector.
	 *
	 * @param cacheKey		缓存key
	 * @param selectorKey	数据操作时用到的key
	 * @param timeout		缓存过期时间
	 */
	public CacheSelector(String cacheKey, String selectorKey, long timeout) {
		this(cacheKey, selectorKey, null, timeout, null);
	}
	
	/**
	 * Creates a new instance of CacheSelector.
	 *
	 * @param cacheKey		缓存key
	 * @param selectorKey	数据操作时用到的key
	 * @param selectorValue 数据操作时传入的value
	 * @param timeout		缓存过期时间
	 */
	public CacheSelector(String cacheKey, String selectorKey, Object selectorValue, 
			long timeout) {
		this(cacheKey, selectorKey, selectorValue, timeout, null);
	}
	
	/**
	 * Creates a new instance of CacheSelector.
	 *
	 * @param cacheKey		缓存key
	 * @param selectorKey	数据操作时用到的key
	 * @param selectorValue 数据操作时传入的value
	 * @param timeout		缓存过期时间
	 * @param emptyValue	数据查询为空时存储缓存的值，如果为null则不存
	 */
	public CacheSelector(String cacheKey, String selectorKey, Object selectorValue,
			long timeout, Object emptyValue) {
		this.cacheKey = cacheKey;
		this.selectorKey = selectorKey;
		this.selectorValue = selectorValue;
		this.timeout = timeout;
		this.emptyValue = emptyValue;
	}

	/**
	 * 获取缓存key
	 * @return  the cacheKey
	 * @since   Ver 1.0
	 */
	public String getCacheKey() {
		return cacheKey;
	}

	/**
	 * 设置缓存key
	 * @param   cacheKey    
	 * @since   Ver 1.0
	 */
	public void setCacheKey(String cacheKey) {
		this.cacheKey = cacheKey;
	}

	/**
	 * 获取数据操作时用到的key
	 * @return  the selectorKey
	 * @since   Ver 1.0
	 */
	public String getSelectorKey() {
		return selectorKey;
	}

	/**
	 * 设置数据操作时用到的key
	 * @param   selectorKey    
	 * @since   Ver 1.0
	 */
	public void setSelectorKey(String selectorKey) {
		this.selectorKey = selectorKey;
	}

	/**
	 * 获取数据操作时传入的value
	 * @return  the selectorValue
	 * @since   Ver 1.0
	 */
	public Object getSelectorValue() {
		return selectorValue;
	}

	/**
	 * 设置数据操作时传入的value
	 * @param   selectorValue    
	 * @since   Ver 1.0
	 */
	public void setSelectorValue(Object selectorValue) {
		this.selectorValue = selectorValue;
	}

	/**
	 * 获取数据过期时间
	 * @return  the timeout
	 * @since   Ver 1.0
	 */
	public long getTimeout() {
		return timeout;
	}

	/**
	 * 设置数据过期时间
	 * @param   timeout    
	 * @since   Ver 1.0
	 */
	public void setTimeout(long timeout) {
		this.timeout = timeout;
	}

	/**
	 * 获取数据查询为空时存储缓存的值，如果为null则不存
	 * @return  the emptyValue
	 * @since   Ver 1.0
	 */
	public Object getEmptyValue() {
		return emptyValue;
	}

	/**
	 * 设置数据查询为空时存储缓存的值，如果为null则不存
	 * @param   emptyValue    
	 * @since   Ver 1.0
	 */
	public void setEmptyValue(Object emptyValue) {
		this.emptyValue = emptyValue;
	}
}

