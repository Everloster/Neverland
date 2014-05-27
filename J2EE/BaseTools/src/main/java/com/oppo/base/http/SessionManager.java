/**
 * OSessionRequestWrapper.java
 * com.oppo.base.http
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-6-29 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 */

package com.oppo.base.http;

import java.io.Serializable;
import java.util.HashMap;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.oppo.base.cache.ICache;
import com.oppo.base.cache.StorageObject;

/**
 * ClassName:OSessionRequestWrapper 
 * Function: 自定义session的request,默认为多机session,需要memcache支持
 * 
 * @author 80036381
 * @version
 * @since Ver 1.1
 * @Date 2011-6-29 下午03:37:18
 */
public class SessionManager {

	private boolean isMultiSession; // 是否是多机Session
	private HttpServletRequest request;
	private HttpServletResponse response;
	private ICache cache;
	private long timeout;
	private long minUpdateTime;

	public SessionManager(HttpServletRequest request, HttpServletResponse response, ICache cache) {
		this(request, response, cache, 30 * 60 * 1000); // 默认为30分钟失效
	}

	public SessionManager(HttpServletRequest request, HttpServletResponse response, ICache cache, long timeout) {
		this(request, response, cache, timeout, 60 * 1000); // 默认为30分钟失效,超过60秒则更新session
	}

	public SessionManager(HttpServletRequest request, HttpServletResponse response, ICache cache, long timeout, long minUpdateTime) {
		this.request = request;
		this.response = response;
		this.cache = cache;
		this.timeout = timeout;
		this.minUpdateTime = minUpdateTime;

		//如果缓存为空或者关闭则不使用多机缓存
		if(null == cache || cache.isTierClosed()) {
			this.isMultiSession = false;
		} else {
			this.isMultiSession = true;
		}
	}

	/**
	 * 获取Session值
	 * 
	 * @param key
	 */
	public Object getSession(String key) {
		if(!this.isMultiSession) {
			return this.request.getSession().getAttribute(key);
		}
		
		String sessionId = getSessionId();
		
		SessionObject sObj = null;
		// 尝试获取session
		try {
			sObj = (SessionObject) cache.get(sessionId, null, 0);
		} catch (Exception e) {
			return null;
		}

		Object t = null;
		if (null != sObj) {
			t = sObj.getSessionValue(key);

			long now = System.currentTimeMillis();
			// 每2分钟重新设置一次session
			if (now - sObj.getLastAccess() >= minUpdateTime) {
				// 重设session最后访问时间
				sObj.setLastAccess(now);
				setCache(sessionId, sObj, timeout);
			}
		}

		return t;
	}

	/**
	 * 设置session值
	 * 
	 * @param key
	 * @param obj
	 */
	public void setSession(String key, Object obj) {
		if(!this.isMultiSession) {
			this.request.getSession().setAttribute(key, obj);
			return;
		}
		
		String sessionId = getSessionId();
		
		SessionObject sObj = null;
		// 尝试获取session
		try {
			sObj = (SessionObject) cache.get(sessionId, null, 0);
		} catch (Exception e) {
		}
		
		//如果不存在则创建
		if(null == sObj) {
			sObj = new SessionObject();
		}
		
		sObj.setSessionValue(key, obj);
		setCache(sessionId, sObj, timeout);
	}

	/**
	 * 移除session值，可以设置是否使用传入的key
	 * 
	 * @param key
	 * @param changeKey
	 *            为true则先转换为session相关的key
	 */
	public void removeSession(String key) {
		if(!this.isMultiSession) {
			this.request.getSession().removeAttribute(key);
			return;
		}

		setSession(key, null);
	}
	
	/**
	 * 获取session id，可重写此方法以完全脱离系统session
	 * @param 
	 * @return
	 */
	protected String getSessionId() {
		return this.request.getSession().getId();
	}

	//设置缓存
	protected void setCache(String key, Object obj, long mSeconds) {
		try {
			cache.insert(key, new StorageObject(key, obj), mSeconds);
		} catch (Exception e) {
		}
	}
	
	/**
	 * 获取是否采用多机session
	 * @return  the isMultiSession
	 * @since   Ver 1.0
	 */
	public boolean isMultiSession() {
		return isMultiSession;
	}

	/**
	 * 设置是否采用多机session
	 * @param   isMultiSession    
	 * @since   Ver 1.0
	 */
	public void setMultiSession(boolean isMultiSession) {
		this.isMultiSession = isMultiSession;
	}

	/**
	 * 获取request
	 * @return  the request
	 * @since   Ver 1.0
	 */
	protected HttpServletRequest getRequest() {
		return request;
	}

	/**
	 * 获取response
	 * @return  the response
	 * @since   Ver 1.0
	 */
	protected HttpServletResponse getResponse() {
		return response;
	}

	/**
	 * Session保存对象
	 * @author   80036381
	 * @version  
	 * @since    Ver 1.1
	 * @Date	 2011-6-30  下午04:08:43
	 * @see
	 */
	private static class SessionObject implements Serializable {
		/**
		 * serialVersionUID:
		 *
		 * @since Ver 1.1
		 */
		private static final long serialVersionUID = 1L;
		
		private long lastAccess;
		private HashMap<String, Object> sessionMap;

		public SessionObject() {
			this.lastAccess = System.currentTimeMillis();
			this.sessionMap = new HashMap<String, Object>();
		}

		/**
		 * 获取最后访问时间
		 * 
		 * @return the lastAccess
		 * @since Ver 1.0
		 */
		public long getLastAccess() {
			return lastAccess;
		}

		/**
		 * 设置最后访问时间
		 * 
		 * @param lastAccess
		 * @since Ver 1.0
		 */
		public void setLastAccess(long lastAccess) {
			this.lastAccess = lastAccess;
		}

		/**
		 * 获取sessionValue
		 * 
		 * @return the key
		 * @since Ver 1.0
		 */
		public Object getSessionValue(String key) {
			return this.sessionMap.get(key);
		}

		/**
		 * 设置sessionValue
		 * 
		 * @param sessionValue
		 * @since Ver 1.0
		 */
		public void setSessionValue(String key, Object sessionValue) {
			if(null == sessionValue) {
				this.sessionMap.remove(key);
			} else {
				this.sessionMap.put(key, sessionValue);
			}
		}
	}

	public static void main(String[] args) {
//		CacheManager cm = CacheManager.getInstance();
//		HashMap<String, String> memcacheConfigMap = new HashMap<String, String>();
//		memcacheConfigMap.put("ServerList", "127.0.0.1:11211");
//
//		ICache cc = null;
//		try {
//			CacheConfig c = new CacheConfig();
//			cm.initCachePool(memcacheConfigMap);
//			cc=cm.getMemcache(null, c);
//		} catch(Exception e) {
//			e.printStackTrace();
//		}
//		
//		SessionManager sessionManager = 
//			new SessionManager(null, cc);
//		sessionManager.setSession("key", "value");
//		System.out.println(sessionManager.getSession("key"));
	}
}





