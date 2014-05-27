/**
 * CachePolicy.java
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

package com.oppo.base.cache.config;
/**
 * ClassName:CachePolicy
 * Function: 缓存策略
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-8  上午11:11:26
 */
public class CachePolicy {
	private boolean isInsertSave;
	private boolean isMemcacheClose;
	private boolean isFileCacheClose;
	private boolean isDBClose;

	public CachePolicy() {
		this(false, false, false);
	}
	
	public CachePolicy(boolean isMemcacheClose, boolean isFileCacheClose, boolean isDBClose) {
		this(isMemcacheClose, isFileCacheClose, isDBClose, false);
	}
	
	public CachePolicy(boolean isMemcacheClose, boolean isFileCacheClose, boolean isDBClose, boolean isInsertSave) {
		this.isMemcacheClose = isMemcacheClose;
		this.isFileCacheClose = isFileCacheClose;
		this.isDBClose = isDBClose;
		this.isInsertSave = isInsertSave();
	}
	
	/**
	 * 插入时是否需要保存
	 * @return
	 */
	public boolean isInsertSave() {
		return isInsertSave;
	}

	public void setInsertSave(boolean isInsertSave) {
		this.isInsertSave = isInsertSave;
	}
	
	/**
	 * memcache是否关闭
	 * @return
	 */
	public boolean isMemcacheClose() {
		return isMemcacheClose;
	}
	
	public void setMemcacheClose(boolean isMemcacheClose) {
		this.isMemcacheClose = isMemcacheClose;
	}
	
	/**
	 * 文件缓存是否关闭
	 * @return
	 */
	public boolean isFileCacheClose() {
		return isFileCacheClose;
	}
	
	public void setFileCacheClose(boolean isFileCacheClose) {
		this.isFileCacheClose = isFileCacheClose;
	}
	
	/**
	 * DB是否关闭
	 * @return
	 */
	public boolean isDBClose() {
		return isDBClose;
	}
	
	public void setDBClose(boolean isDBClose) {
		this.isDBClose = isDBClose;
	}
}

