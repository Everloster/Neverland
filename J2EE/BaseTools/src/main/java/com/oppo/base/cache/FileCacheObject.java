/**
 * FileCacheObject.java
 * com.oppo.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-19 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache;
/**
 * ClassName:FileCacheObject
 * Function: 文件缓存对象
 * 主要是为了防止文件缓存到期后无法判断是否需要删除的情况
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-8-19  上午09:22:40
 */
public class FileCacheObject implements java.io.Serializable {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	private long createTime;
	private long timeout;			
	private Object storageObject;
	
	public FileCacheObject() {
		
	}
	
	/**
	 * 
	 * Creates a new instance of FileCacheObject.
	 * @param timeout		过期时间(秒)
	 * @param storageObject
	 */
	public FileCacheObject(long timeout, Object storageObject) {
		this(System.currentTimeMillis(), timeout, storageObject);
	}
	
	/**
	 * 
	 * Creates a new instance of FileCacheObject.
	 * @param createTime	创建时间(毫秒)
	 * @param timeout		过期时间(秒)
	 * @param storageObject
	 */
	public FileCacheObject(long createTime, long timeout, Object storageObject) {
		this.createTime = createTime;
		this.timeout = timeout;
		this.storageObject = storageObject;
	}
	
	/**
	 * 获取缓存创建时间
	 * @return  the createTime
	 * @since   Ver 1.0
	 */
	public long getCreateTime() {
		return createTime;
	}
	/**
	 * 设置缓存创建时间
	 * @param   createTime    
	 * @since   Ver 1.0
	 */
	public void setCreateTime(long createTime) {
		this.createTime = createTime;
	}
	/**
	 * 获取缓存过期时间
	 * @return  the timeout
	 * @since   Ver 1.0
	 */
	public long getTimeout() {
		return timeout;
	}
	/**
	 * 设置缓存过期时间
	 * @param   timeout    
	 * @since   Ver 1.0
	 */
	public void setTimeout(long timeout) {
		this.timeout = timeout;
	}
	/**
	 * 获取storageObject
	 * @return  the storageObject
	 * @since   Ver 1.0
	 */
	public Object getStorageObject() {
		return storageObject;
	}
	/**
	 * 设置storageObject
	 * @param   storageObject    
	 * @since   Ver 1.0
	 */
	public void setStorageObject(Object storageObject) {
		this.storageObject = storageObject;
	}
	
	/**
	 * 缓存是否过期
	 * @param 
	 * @return
	 */
	public boolean isCacheExpire() {
		return isCacheExpire(System.currentTimeMillis());
	}
	
	/**
	 * 缓存是否过期
	 * @param 
	 * @return
	 */
	public boolean isCacheExpire(long inputTime) {
		//等于0表示无过期时间
		if(timeout == 0) {
			return false;
		}
		
		return createTime + timeout < inputTime;
	}
}

