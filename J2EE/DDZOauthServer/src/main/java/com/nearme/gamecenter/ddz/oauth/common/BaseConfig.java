/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年1月7日
 */
package com.nearme.gamecenter.ddz.oauth.common;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * 
 * @Author LaiLong
 * @Since 2014年1月7日
 */
public class BaseConfig {

	private static Logger logger = LoggerFactory.getLogger(BaseConfig.class);

	private static final String PROJECT_CONFIG = "/project.xml";

	private static final Map<String, Object> INPUTSTREAM_CONFIG_MAP = new ConcurrentHashMap<String, Object>();

	private static final Map<String, String> PROJECT_CONFIG_MAP = getConfigMapFromPath(PROJECT_CONFIG);
	
	public static final String APP_KEY = getProjectConfig("app-key");
	
	public static final String APP_SECRET = getProjectConfig("app-secret");

	public static String getProjectConfig(String configSec) {
		return PROJECT_CONFIG_MAP.get(configSec);
	}

	@SuppressWarnings("unchecked")
	private static ConcurrentHashMap<String, String> getConfigMapFromPath(
			String path) {
		Object obj = INPUTSTREAM_CONFIG_MAP.get(path);
		if (obj == null) {
			logger.info("start to load config file : " + path);
			final Map<String, String> map = parseSimpleXmlInputStreamToMap(getInputStreamWithClass(path));
			if (map != null) {
				obj = new ConcurrentHashMap<String, String>(map);
				INPUTSTREAM_CONFIG_MAP.put(path, obj);
			}
		}
		return (ConcurrentHashMap<String, String>) obj;
	}

	private static InputStream getInputStreamWithClass(String path) {
		return BaseConfig.class.getResourceAsStream(path);
	}

	private static Map<String, String> parseSimpleXmlInputStreamToMap(
			InputStream stream) {
		Map<String, String> map = new HashMap<String, String>();
		try {
			final DocumentBuilderFactory builderFactory = DocumentBuilderFactory
					.newInstance();
			final DocumentBuilder builder = builderFactory.newDocumentBuilder();
			final Document document = builder.parse(stream);

			final Element rootElement = document.getDocumentElement();

			final NodeList nodes = rootElement.getChildNodes();
			for (int i = 0; i < nodes.getLength(); i++) {
				final Node node = nodes.item(i);
				if (node.getNodeType() == Node.ELEMENT_NODE) {
					map.put(node.getNodeName(), node.getTextContent());
				}
			}
		} catch (ParserConfigurationException e) {
			e.printStackTrace();
		} catch (SAXException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return map;
	}

}
