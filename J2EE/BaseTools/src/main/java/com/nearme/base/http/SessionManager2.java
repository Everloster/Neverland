/**
 * SessionManager2.java
 * com.nearme.base.http
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-11-22 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.http;

import java.io.Serializable;
import java.util.HashMap;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.nearme.base.cache.ICache2;
import com.nearme.base.cache.selector.CacheSelector;

/**
 * ClassName:SessionManager2 <br>
 * Function: 自定义session的request,默认为多机session,需要memcache支持 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-11-22  上午10:33:05
 */
public class SessionManager2 {
	/**
	 * 默认为30分钟失效
	 */
	public static final long DEFAULT_TIMEOUT = 30 * 60 * 1000;
	
	/**
	 * 超过60秒则更新session
	 */
	public static final long DEFAULT_UPDATETIME = 60 * 1000;
	
	private boolean isMultiSession; // 是否是多机Session
	private ICache2 cache;
	private long timeout;
	private long minUpdateTime;

	public SessionManager2(ICache2 cache) {
		this(cache, DEFAULT_TIMEOUT); // 默认为30分钟失效
	}

	public SessionManager2(ICache2 cache, long timeout) {
		this(cache, timeout, DEFAULT_UPDATETIME); // 默认为30分钟失效,超过60秒则更新session
	}

	public SessionManager2(ICache2 cache, long timeout, long minUpdateTime) {
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
	public Object getSession(HttpServletRequest request, HttpServletResponse response, String key) {
		if(!this.isMultiSession) {
			return request.getSession().getAttribute(key);
		}
		
		String sessionId = getSessionId(request, response);
		
		SessionObject sObj = null;
		// 尝试获取session
		try {
			sObj = (SessionObject) cache.get(new CacheSelector(sessionId, null, 0));
		} catch (Exception e) {
			return null;
		}

		Object t = null;
		if (null != sObj) {
			t = sObj.getSessionValue(key);

			long now = System.currentTimeMillis();
			// 每隔指定时间重新设置一次session
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
	public void setSession(HttpServletRequest request, HttpServletResponse response, String key, Object obj) {
		if(!this.isMultiSession) {
			request.getSession().setAttribute(key, obj);
			return;
		}
		
		String sessionId = getSessionId(request, response);
		
		SessionObject sObj = null;
		// 尝试获取session
		try {
			sObj = (SessionObject) cache.get(new CacheSelector(sessionId, null, 0));
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
	public void removeSession(HttpServletRequest request, HttpServletResponse response, String key) {
		if(!this.isMultiSession) {
			request.getSession().removeAttribute(key);
			return;
		}

		setSession(request, response, key, null);
	}
	
	/**
	 * 获取session id，可重写此方法以完全脱离系统session
	 * @param 
	 * @return
	 */
	protected String getSessionId(HttpServletRequest request, HttpServletResponse response) {
		return request.getSession().getId();
	}

	//设置缓存
	protected void setCache(String key, Object obj, long mSeconds) {
		try {
			cache.insert(new CacheSelector(key, null, obj, mSeconds));
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
}

