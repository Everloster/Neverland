/**
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 * FileName:NodeParse.java
 * ProjectName:OStat_Record
 * PackageName:com.oppo.statistic.xml
 * Create Date:2011-5-3
 * History:
 *   ver	date	  author		desc	
 * ────────────────────────────────────────────────────────
 *   1.0	2011-5-3	  80036381		
 *
 * 
*/

package com.oppo.base.xml;

import java.lang.reflect.Method;

import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;

import com.oppo.base.beanutils.BeanUtil;
import com.oppo.base.beanutils.DefaultNameTransfer;
import com.oppo.base.beanutils.INameTransfer;
import com.oppo.base.beanutils.IValueTransfer;
import com.oppo.base.beanutils.StringValueTransfer;

/**
 * ClassName:NodeParse
 * Function: 将一个独立的xml节点解析为对象的工具
 * 节点示例：<Item name="itemName" value="itemValue" />
 * 
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-5-3  上午08:00:50
 */
public class XmlNodeParse {
	/**
	 * 从xml的单个节点中的属性获取实体
	 * @param node xml需要解析的节点
	 * @param tClass 需要解析成的节点的类型
	 * @return
	 */
	public static <T> T getObjectFromXmlNode(Node node, Class<T> tClass) throws Exception {
		return getObjectFromXmlNode(node, tClass, new DefaultNameTransfer());
	}
	
	/**
	 * 从xml的单个节点中的属性获取实体
	 * @param node xml需要解析的节点
	 * @param tClass 需要解析成的节点的类型
	 * @param nameTrans xml节点与实体属性名称装换
	 * @return
	 */
	public static <T> T getObjectFromXmlNode(Node node, Class<T> tClass, INameTransfer nameTrans) throws Exception {
		NamedNodeMap nodeMap = node.getAttributes();
		int attrLen = nodeMap.getLength();
		if(attrLen < 1) {
			return null;
		}
		
		//如果未提供，则使用默认
        if(null == nameTrans) {
        	nameTrans = new DefaultNameTransfer();
        }
        
        IValueTransfer<String> valueTransfer = new StringValueTransfer();
		
		T tObj = tClass.newInstance();
		Method[] methods = tClass.getMethods();
		for(int i = 0; i < attrLen; i++) {
			//属性节点
			Node attributeNode = nodeMap.item(i);
			String name = nameTrans.transferFromNodeName(attributeNode.getNodeName());
			String value = attributeNode.getNodeValue();
			
			BeanUtil.setValueToObject(nameTrans, valueTransfer, tObj, methods, name, value);
		}
		
		return tObj;
	}
}

