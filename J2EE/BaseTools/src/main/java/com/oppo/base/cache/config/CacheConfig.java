/**
 * CacheConfig.java
 * com.oppo.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-6-30 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache.config;

import org.apache.ibatis.session.SqlSessionFactory;

/**
 * ClassName:CacheConfig
 * Function: 3层缓存的配置
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-30  上午10:51:29
 */
public class CacheConfig {
	private String memcachePoolName;
	private String fileCacheDir;
	private SqlSessionFactory sqlWriteFactory;
	private SqlSessionFactory sqlReadFactory;
	private CachePolicy cachePolicy;
	
	/**
	 * 获取memcache池名
	 * @return  the memcachePoolName
	 * @since   Ver 1.0
	 */
	public String getMemcachePoolName() {
		return memcachePoolName;
	}
	
	/**
	 * 设置memcache池名
	 * @param   memcachePoolName    
	 * @since   Ver 1.0
	 */
	public void setMemcachePoolName(String memcachePoolName) {
		this.memcachePoolName = memcachePoolName;
	}
	
	/**
	 * 获取文件缓存地址
	 * @return  the fileCacheDir
	 * @since   Ver 1.0
	 */
	public String getFileCacheDir() {
		return fileCacheDir;
	}
	
	/**
	 * 设置文件缓存地址
	 * @param   fileCacheDir    
	 * @since   Ver 1.0
	 */
	public void setFileCacheDir(String fileCacheDir) {
		this.fileCacheDir = fileCacheDir;
	}
	
	/**
	 * 获取数据写Session
	 * @return  the sqlWriteSession
	 * @since   Ver 1.0
	 */
	public SqlSessionFactory getSqlWriteFactory() {
		return sqlWriteFactory;
	}
	
	/**
	 * 设置数据写Session
	 * @param   sqlWriteSession    
	 * @since   Ver 1.0
	 */
	public void setSqlWriteFactory(SqlSessionFactory sqlWriteFactory) {
		this.sqlWriteFactory = sqlWriteFactory;
	}
	
	/**
	 * 获取数据读Session
	 * @return  the sqlReadSession
	 * @since   Ver 1.0
	 */
	public SqlSessionFactory getSqlReadFactory() {
		return sqlReadFactory;
	}
	
	/**
	 * 设置数据读Session
	 * @param   sqlReadSession    
	 * @since   Ver 1.0
	 */
	public void setSqlReadFactory(SqlSessionFactory sqlReadFactory) {
		this.sqlReadFactory = sqlReadFactory;
	}
	
	/**
	 * 获取缓存策略
	 * @return  the cachePolicy
	 * @since   Ver 1.0
	 */
	public CachePolicy getCachePolicy() {
		if(null == cachePolicy) {
			cachePolicy = new CachePolicy();
		}
		
		return cachePolicy;
	}
	
	/**
	 * 设置缓存策略
	 * @param   cachePolicy    
	 * @since   Ver 1.0
	 */
	public void setCachePolicy(CachePolicy cachePolicy) {
		this.cachePolicy = cachePolicy;
	}
}

