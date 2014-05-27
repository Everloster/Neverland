/**
 * AbstractMultiSession.java
 * com.nearme.base.http
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-12-2 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.http;

import java.util.HashMap;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.nearme.base.cache.ICache2;
import com.nearme.base.cache.Memcache2;
import com.nearme.base.cache.util.CacheManager2;
import com.oppo.base.cache.config.CacheConfig;
import com.oppo.base.common.NumericUtil;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:AbstractMultiSession <br>
 * Function: 多机session操作类,实现类应该采取单例模式 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-12-2  上午11:33:30
 */
public abstract class AbstractMultiSession {
	/**
	 * 多机session对应的服务器列表
	 */
	public static final String CONFIG_SESSION_SERVERLIST = "session-serverlist";
	
	/**
	 * 多机session对应的缓存池名
	 */
	public static final String CONFIG_SESSION_POOLNAME = "session-poolname";
	
	/**
	 * 是否多机session的配置项
	 */
	public static final String CONFIG_SESSION_MULTI = "session-multi";
	
	/**
	 * 多机session缓存过期时间
	 */
	public static final String CONFIG_SESSION_TIMEOUT = "session-timeout";
	
	/**
	 * session在缓存中的前缀配置项，如果设置的值为空则随机分配
	 */
	public static final String CONFIG_SESSION_PREFIX = "session-prefix";
	
	private SessionManager2 sessionManager;
	
	public AbstractMultiSession(){
		initial();
	}
	
	/**
	 * 初始化多机session使用的缓存的设置
	 * @param 
	 * @return
	 */
	protected void initial() {
		//memcache设置
		//读取配置内容
		Map<String, String> configMap = getMultiSessionConfig();
		//是否是多机session
		boolean isMultiSession = Boolean.parseBoolean(configMap.get(CONFIG_SESSION_MULTI));
		//session过期时间
		long sessionTimeout = NumericUtil.parseLong(configMap.get(CONFIG_SESSION_TIMEOUT), 
				SessionManager2.DEFAULT_TIMEOUT);
		
		ICache2 cache = null;
		if(isMultiSession) { //多机缓存时初始化cache
			//缓存服务器列表
			String serverList = configMap.get(CONFIG_SESSION_SERVERLIST);
			//缓存池名称
			String poolName = configMap.get(CONFIG_SESSION_POOLNAME);

			//memcache设置
			HashMap<String, String> memcacheConfigMap = new HashMap<String, String>(2);
			memcacheConfigMap.put(Memcache2.CONFIG_SERVER_LIST, serverList);
			memcacheConfigMap.put(Memcache2.CONFIG_POOL_NAME, poolName);
	
			CacheConfig cacheConfig = new CacheConfig();
			cacheConfig.setMemcachePoolName(poolName);
			try {
				CacheManager2 cm = CacheManager2.getInstance();
				cm.initCachePool(memcacheConfigMap);
				
				//设置session的缓存key前缀
				String sessionPrefix = configMap.get(CONFIG_SESSION_PREFIX);
				if(StringUtil.isNullOrEmpty(sessionPrefix)) {
					sessionPrefix = StringUtil.generateRandomString(7);
				}
				
				cache = cm.getMemcache(sessionPrefix, cacheConfig);
			} catch(Exception e) {
				error("多机session配置初始化失败", e);
			}
		}
		
		sessionManager = new SessionManager2(cache, sessionTimeout);
		sessionManager.setMultiSession(isMultiSession);
	}
	
	/**
	 * 获取多机session配置 <br>
	 * session-serverlist、session-poolname字段为必选(memcache配置) <br>
	 * session-timeout为可选，session过期时间，默认为30分钟 <br>
	 * session-prefix为可选，session缓存的key前缀，不填则随机分配 <br>
	 * session-multi为可选，是否采取多机缓存，默认为false <br>
	 * @param 
	 * @return
	 */
	public abstract Map<String, String> getMultiSessionConfig();
	
	/**
	 * 获取session值
	 * @param 
	 * @return
	 */
	public Object get(HttpServletRequest request, HttpServletResponse response, String key) {
		SessionManager2 sessionManager = getSessionManager();
		return sessionManager.getSession(request, response, key);
	}
	
	/**
	 * 
	 * 设置session值
	 * @param 
	 * @return
	 */
	public void set(HttpServletRequest request, HttpServletResponse response, String key, Object value) {
		SessionManager2 sessionManager = getSessionManager();
		sessionManager.setSession(request, response, key, value);
	}
	
	/**
	 * 移除session值
	 * @param 
	 * @return
	 */
	public void remove(HttpServletRequest request, HttpServletResponse response, String key) {
		SessionManager2 sessionManager = getSessionManager();
		sessionManager.removeSession(request, response, key);
	}
	
	/**
	 * 记录错误日志
	 * @param 
	 * @return
	 */
	protected void error(String errorMsg, Throwable throwable) {
		
	}
	
	protected SessionManager2 getSessionManager() {
		return sessionManager;
	}
}

