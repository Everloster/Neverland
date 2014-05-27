/**
 * DefaultBusinessFactory.java
 * com.nearme.base.netty.server
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-31 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.server;

import java.io.InputStream;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nearme.base.netty.common.RequestConfig;
import com.oppo.base.xml.XmlSimpleListParse;

/**
 * ClassName:DefaultBusinessFactory <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-31  下午5:44:06
 */
public class DefaultBusinessFactory implements IBusinessFactory {

	private static Logger logger = LoggerFactory.getLogger(NettyBusinessFactory.class);

	private final ConcurrentHashMap<Integer, IBusinessHandler> businessMap =
		new ConcurrentHashMap<Integer, IBusinessHandler>();

	public DefaultBusinessFactory() {

	}

	public DefaultBusinessFactory(String webXmlPath) {
		if (null == webXmlPath) {
			webXmlPath = "/web.xml";
		}

		registerXml(this.getClass().getResourceAsStream(webXmlPath));
	}

	@Override
	public boolean registerXml(InputStream inputStream) {
		XmlSimpleListParse<RequestConfig> xselp = new XmlSimpleListParse<RequestConfig>();
		xselp.setEntityClass(RequestConfig.class);
		xselp.readFromFile(inputStream);

		//将保存配置项
		List<RequestConfig> list = xselp.getObjectList();
		if(null == list) {
			return false;
		}

		boolean registerOnce = false;
		for(int i = 0, size = list.size(); i < size; i++) {
			RequestConfig reqConf = list.get(i);
			int businessType = Integer.parseInt(reqConf.getExtension0());
			String handleClass = reqConf.getHandleClass();
			//如果还不存在则尝试新建
			try {
				Class<?> cls = Class.forName(handleClass);
				Object obj = cls.newInstance();

				IBusinessHandler handler;
				if(obj instanceof IBusinessHandler) {
					handler = (IBusinessHandler)obj;
					register(businessType, handler);

					registerOnce = true;
				} else {
					logger.error("handle class " + handleClass + " is not a request handler class:" + obj.getClass().getName());
				}
			} catch (ClassNotFoundException e) {
				logger.error("handle class " + handleClass + " not found!");
			} catch (InstantiationException e) {
				logger.error("handle class " + handleClass + " create fail!");
			} catch (IllegalAccessException e) {
				logger.error("handle class " + handleClass + " cannot be access!");
			}
		}

		return registerOnce;
	}

	@Override
	public void register(int businessType, IBusinessHandler handler) {
		businessMap.putIfAbsent(businessType, handler);
	}

	@Override
	public IBusinessHandler getBusinessHandler(int businessType) {
		return businessMap.get(businessType);
	}
}

