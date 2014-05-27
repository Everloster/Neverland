/**
 * CacheObject.java
 * com.oppo.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-6-8 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache;
/**
 * ClassName:CacheObject
 * Function: 存储的实体
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-8  上午10:58:29
 */
public class StorageObject implements java.io.Serializable {
	/**
	 * serialVersionUID:
	 *
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	private String storageId;
	private Object storageObject;
	/**
	 * 存储对应id
	 * @return  the storageId
	 * @since   Ver 1.0
	 */
	public String getStorageId() {
		return storageId;
	}
	/**
	 * 存储对应id
	 * @param   storageId    
	 * @since   Ver 1.0
	 */
	public void setStorageId(String storageId) {
		this.storageId = storageId;
	}
	/**
	 * 获取存储对象
	 * @return  the storageObject
	 * @since   Ver 1.0
	 */
	public Object getStorageObject() {
		return storageObject;
	}
	/**
	 * 设置存储对象
	 * @param   storageObject    
	 * @since   Ver 1.0
	 */
	public void setStorageObject(Object storageObject) {
		this.storageObject = storageObject;
	}
	
	public StorageObject(String cacheId, Object cacheObject) {
		this.storageId = cacheId;
		this.storageObject = cacheObject;
	}
}

