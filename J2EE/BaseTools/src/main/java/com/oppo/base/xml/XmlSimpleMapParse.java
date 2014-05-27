package com.oppo.base.xml;

import java.util.HashMap;

import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;

/**
 * Xml文件信息读取，将xml中的节点解析为key-value对
 * 例如:节点<Cacheable>true</Cacheable>解析到的key-value为：Cacheable -> true
 * @author yangbo
 * 2011-1-7
 */
public class XmlSimpleMapParse extends XmlSimpleParse {

	private HashMap<String, String> xmlNodeMap;
	
	public HashMap<String, String> getXmlMap() {
		if(null == xmlNodeMap) {
			xmlNodeMap = new HashMap<String, String>();
		}
		
		return xmlNodeMap;
	}

	/**
	 * 遍历节点,记录所有文本节点
	 * @param node
	 */
	protected void visitNode(Node node) {
		NodeList nList = node.getChildNodes();
		int len = nList.getLength();
		for(int i = 0; i < len; i++) {
			Node childNode = nList.item(i);
			//将Text节点作为属性值
			if(childNode instanceof Text) {
				String text = ((Text)childNode).getWholeText().trim();
				if(1 == len && !text.equals("")) {
					xmlNodeMap.put(node.getNodeName(), text);
				}
			} else {
				visitNode(childNode);
			}
		}
	}
	
	/**
	 * 清除已解析到的节点信息
	 */
	public void reset() {
		if(null == xmlNodeMap) {
			xmlNodeMap = new HashMap<String, String>();
		} else {
			xmlNodeMap.clear();
		}
	}
}
