/**
 * XmlSimpleListParse.java
 * com.oppo.base.xml
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-9-26 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.xml;

import java.util.ArrayList;
import java.util.List;

import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * ClassName:XmlSimpleListParse
 * Function: 将xml中根节点下的所有第一级子节点解析为实体list
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-26  下午01:06:21
 */
public class XmlSimpleListParse<T> extends XmlSimpleParse {

	private List<T> objectList;
	private Class<T> entityClass;
	
	public XmlSimpleListParse() {
	}
	
	public XmlSimpleListParse(Class<T> entityClass) {
		this.entityClass = entityClass;
	}

	/**
	 * 遍历节点,记录所有文本节点
	 * @param node
	 */
	protected void visitNode(Node node) throws Exception {
		NodeList nList = node.getChildNodes();
		int len = nList.getLength();
		
		XmlObjectParser parser = new XmlObjectParser();
		for(int i = 0; i < len; i++) {
			Node childNode = nList.item(i);

			
			T obj = parser.parseXmlNode(childNode, entityClass);
			if(null != obj) {
				objectList.add(obj);
			}
		}
	}
	
	/**
	 * 清除已解析到的节点信息
	 */
	public void reset() {
		if(null == objectList) {
			objectList = new ArrayList<T>();
		} else {
			objectList.clear();
		}
	}
	
	public List<T> getObjectList() {
		return objectList;
	}

	/**
	 * 获取实体的class
	 * @return  the entityClass
	 * @since   Ver 1.0
	 */
	public Class<?> getEntityClass() {
		return entityClass;
	}

	/**
	 * 设置实体的class
	 * @param   entityClass    
	 * @since   Ver 1.0
	 */
	public void setEntityClass(Class<T> entityClass) {
		this.entityClass = entityClass;
	}
}

