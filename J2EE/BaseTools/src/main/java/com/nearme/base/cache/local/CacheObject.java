/**
 * CacheObject.java
 * com.nearme.base.cache.local
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-27 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache.local;
/**
 * ClassName:CacheObject <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-27  下午04:36:08
 */
public class CacheObject implements java.io.Serializable {
	private static final long serialVersionUID = 1L;
	
	private long initTime;		//缓存的初始化时间
	private long expireTime;	//缓存的过期时间
	private Object key;			//缓存键
	private Object value;		//缓存值
	private static final long MAX_EXPIRE_TIME = 30 * 24 * 3600 * 1000L; 
	
	public CacheObject() {
	}
	
	/**
	 * 
	 * Creates a new instance of CacheObject.
	 *
	 * @param key 缓存键
	 * @param value 缓存值
	 * @param cacheTime 缓存过期时间
	 */
	public CacheObject(Object key, Object value, long cacheTime) {
		this.key = key;
		this.value = value;
		
		this.initTime = System.currentTimeMillis();
		computeExpireTime(cacheTime);
	}
	
	/**
	 * 缓存是否已经过期
	 * @param 
	 * @return
	 */
	public boolean isCacheExpire() {
		return System.currentTimeMillis() > this.expireTime;
	}
	
	//计算过期时间
	private void computeExpireTime(long cacheTime) {
		//当缓存时间小于0或者大于最大缓存时间时，缓存时间设置为最长缓存时间
		if(cacheTime < 0 || cacheTime > MAX_EXPIRE_TIME) {
			cacheTime = MAX_EXPIRE_TIME;
		}
		
		this.expireTime = this.initTime + cacheTime;
	}

	/**
	 * 获取缓存初始化的时间
	 * @param 
	 * @return
	 */
	public long getInitTime() {
		return initTime;
	}

	/**
	 * 获取过期时间
	 * @param 
	 * @return
	 */
	public long getExpireTime() {
		return expireTime;
	}

	/**
	 * 设置过期时间
	 * @param expireTime
	 * @return
	 */
	public void setExpireTime(long expireTime) {
		this.expireTime = expireTime;
	}

	/**
	 * 获取缓存键
	 * @param 
	 * @return
	 */
	public Object getKey() {
		return key;
	}

	/**
	 * 设置缓存键
	 * @param 
	 * @return
	 */
	public void setKey(Object key) {
		this.key = key;
	}

	/**
	 * 获取缓存值
	 * Function Description here
	 * @param 
	 * @return
	 */
	public Object getValue() {
		return value;
	}

	/**
	 * 设置缓存值
	 * @param 
	 * @return
	 */
	public void setValue(Object value) {
		this.value = value;
	}
}
