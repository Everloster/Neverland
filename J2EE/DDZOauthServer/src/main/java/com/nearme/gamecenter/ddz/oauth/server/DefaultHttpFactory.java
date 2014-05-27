/**
 * DefaultHttpFactory.java
 * com.nearme.base.netty.server
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-29 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.gamecenter.ddz.oauth.server;

import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nearme.gamecenter.ddz.oauth.common.RequestConfig;
import com.oppo.base.xml.XmlSimpleListParse;

/**
 * ClassName:DefaultHttpFactory <br>
 * Function: 请求处理工厂，注意web.xml目录必须与class所在的根目录相同 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-1-29  下午01:52:06
 */
public class DefaultHttpFactory implements IHttpFactory {
	private static final String WEB_CONFIG_PATH = "/web.xml";
	private static Map<String, RequestConfig> REQUEST_CONFIG_MAP;
	private static ConcurrentHashMap<String, IRequestHandler> REQUEST_HANDLER_MAP;
	private static Logger logger = LoggerFactory.getLogger(DefaultHttpFactory.class);

	private static final DefaultHttpFactory DEFAULT_HTTP_FACTORY = new DefaultHttpFactory(WEB_CONFIG_PATH);

	public DefaultHttpFactory() {
	}

	public DefaultHttpFactory(String webXmlPath) {
		if (null == webXmlPath) {
			webXmlPath = WEB_CONFIG_PATH;
		}

		this.registerXml(this.getClass().getResourceAsStream(webXmlPath));
	}

	public static DefaultHttpFactory getDefaultHttpFactory() {
		return DEFAULT_HTTP_FACTORY;
	}

	/**
	 * 根据传入的url获取处理的实例
	 * @param requestUri url
	 * @return
	 */
	public IRequestHandler getRequestHandler(String requestUri) {
		if(null == requestUri) {
			return null;
		}

		int index = requestUri.indexOf('?');
		if(index >= 0) {
			requestUri = requestUri.substring(0, index);
		}

		//找出对应的配置项
		RequestConfig config = REQUEST_CONFIG_MAP.get(requestUri);
		if(null == config) {
			logger.warn("request url " + requestUri + " is not configured!");
			return null;
		}

		String className = config.getHandleClass();
		IRequestHandler handler = REQUEST_HANDLER_MAP.get(requestUri);
		if(null == handler) {
			//如果还不存在则尝试新建
			try {
				Class<?> cls = Class.forName(className);
				Object obj = cls.newInstance();
				if(obj instanceof IRequestHandler) {
					handler = (IRequestHandler)obj;
					handler.init(config);
					//创建成功后放入map中
					IRequestHandler existHandler = REQUEST_HANDLER_MAP.putIfAbsent(requestUri, handler);
					if(null != existHandler) {
						handler = existHandler;
					}
				} else {
					logger.error("request url " + requestUri + " configured with class name " + className + " is not a request handler class!");
				}
			} catch (ClassNotFoundException e) {
				logger.error("request url " + requestUri + " configured with class name " + className + " not found!");
			} catch (InstantiationException e) {
				logger.error("request url " + requestUri + " configured with class name " + className + " create fail!");
			} catch (IllegalAccessException e) {
				logger.error("request url " + requestUri + " configured with class name " + className + " cannot be access!");
			}
		}

		return handler;
	}

	@Override
	public boolean registerXml(InputStream inputStream) {
		REQUEST_CONFIG_MAP = new ConcurrentHashMap<String, RequestConfig>();
		REQUEST_HANDLER_MAP = new ConcurrentHashMap<String, IRequestHandler>();

		//读取web.xml配置文件
		XmlSimpleListParse<RequestConfig> xselp = new XmlSimpleListParse<RequestConfig>();
    	xselp.setEntityClass(RequestConfig.class);
    	xselp.readFromFile(inputStream);

    	//将保存配置项
    	List<RequestConfig> list = xselp.getObjectList();
    	for(RequestConfig config : list) {
    		REQUEST_CONFIG_MAP.put(config.getRequestUrl(), config);
    	}

    	list.clear();
    	list = null;

    	return true;
	}
}

