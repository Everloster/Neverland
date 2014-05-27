/**
 * XmlListParser.java
 * com.oppo.base.xml
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-30 		80054252
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.xml;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import com.oppo.base.beanutils.MethodsUtil;

/**
 * ClassName:XmlListParser
 * Function: 将对象解析成XML，如果对象中包含List类型的属性，则在解析前需要设置对应的class
 * 调用方式为：
 * XmlListParser listParser = new XmlListParser();
 * listParser.setListClass(YourList.class);
 * YourObject yourObj = listParser.getObjectFromXml("<...your xml...>", YourObject.class);
 *
 * @author   80054252
 * @version  
 * @since    Ver 1.0
 * @Date	 2011-8-30
 */
public class XmlListParser extends XmlObjectParser {
	private Class<?> listClass;
	private String subListName;
	
	public XmlListParser() {
		super();
	}
	
	public XmlListParser(Class<?> listClass, String subListName) {
		super();
		this.listClass = listClass;
		this.subListName = subListName;
	}

	/**
	 * 获取subListName
	 * @return  the subListName
	 * @since   Ver 1.0
	 */
	public String getSubListName() {
		if(null == subListName) {
			return "ele";
		}
		return subListName;
	}


	/**
	 * 设置subListName
	 * @param   subListName    
	 * @since   Ver 1.0
	 */
	public void setSubListName(String subListName) {
		this.subListName = subListName;
	}


	/**
	 * 获取实体中list属性中包含的class
	 * @return  the listClass
	 * @since   Ver 1.0
	 */
	public Class<?> getListClass() {
		return listClass;
	}


	/**
	 * 设置实体中list属性中包含的class
	 * @param   listClass    
	 * @since   Ver 1.0
	 */
	public void setListClass(Class<?> listClass) {
		this.listClass = listClass;
	}


	/**
     * 处理非text类型的子节点
     * @param 
     * @return
     */
    protected <T> boolean handleNotTextNode(Node parentNode, NodeList subNodeList, T tObj) throws Exception {
    	if(null == listClass) {
    		return false;
    	}

    	int subNodeCount = subNodeList.getLength();
    	
    	List<Object> dataList = new ArrayList<Object>(subNodeCount);
        for (int j = 0; j < subNodeCount; j++) {
            Node gSubNode = subNodeList.item(j);
            Object obj = super.parseXmlNode(gSubNode, listClass);
            if (null != obj) {
                dataList.add(obj);
            }
        }

        if (dataList.size() > 0) {
        	String setMethodName = this.getNameTransfer().getSetMethod(parentNode.getNodeName());
        	Method m = MethodsUtil.getMethod(tObj.getClass(), setMethodName);
        	m.invoke(tObj, dataList);
        	
        	return true;
        } else {
        	return false;
        }
    }
	
	
	/**
	 * 根据指定的类型解析xml
	 * @param xmlText 包含数据的XML
	 * @param tClass 数据实体类型
	 */
    public <T> T getListFromXml(String xmlText, Class<T> tClass, List<T> list) throws Exception {
    	
    	return getListFromXml(new ByteArrayInputStream(xmlText.getBytes(getEncoding())), tClass, list);
    }
    
    /**
	 * 根据指定的类型解析xml
	 * @param xmlText 包含数据的XML
	 * @param tClass 数据实体类型
	 */
    public <T> T getListFromXml(InputStream xmlStream, Class<T> tClass, List<T> list) throws Exception {
    	clearState();
    	
    	DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();
		Document doc = db.parse(xmlStream);
		Node node = doc.getDocumentElement();
		NodeList objList = node.getChildNodes();
		int objCount = objList.getLength();
    	for(int index = 0; index < objCount; index++) {
    		Node objNode = objList.item(index);
    		if(checkRootNode(objNode)) {
    			T obj = parseXmlNode(objNode, tClass);
    			if(obj != null) {
    				list.add(obj);
    			}
    		} else {
    			throw new Exception("root node error");
    		}
    	}
    	
    	return null;
    }
	
	/**
     * 利用指定List对象生成xml
     * @param objList 需要解析的List
     * @param documentName 根节点
     * @param needEmpty 是否加上属性为空的节点
     * @return
     */
    public String generateListXML(List<Object> objList, String documentName) throws Exception {
    	return generateListXML(objList, documentName, false);
    }
    
    /**
     * 利用指定List对象生成xml
     * @param objList 需要解析的List
     * @param documentName 根节点
     * @param needEmpty 是否加上属性为空的节点
     * @return
     */
    public String generateListXML(List<Object> objList, String documentName, boolean needEmpty) throws Exception {
    	clearState();
    	
    	if (objList != null && objList.size() > 0) {
        	//生成document
        	DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        	DocumentBuilder db = dbf.newDocumentBuilder();
			Document doc = db.newDocument();
			Element ele = doc.createElement(documentName);
			
        	for(Object obj : objList) {
        		// 取得类名
        		String className = obj.getClass().getName();
        		className = className.substring(className.lastIndexOf('.') + 1);
        		// 生成类节点
        		Element classNode = doc.createElement(className);
        		
        		//逐个解析obj中的方法
            	Method[] ms = MethodsUtil.getMethods(obj.getClass());
            	int mLen = ms.length;
                for (int i = 0; i < mLen; i++) {
                	Method method = ms[i];
                	if(!Modifier.isPublic(method.getModifiers())) {
                		continue;
                	}
                	
                	String methodName = method.getName();
                	if(methodName.startsWith("get") && !methodName.equals("getClass")) {
                		String nodeName = getNameTransfer().transferToNodeName(methodName);
                		Object value = method.invoke(obj);
                        Node childNode = getPropertyValue(doc, nodeName, value, needEmpty, null, null);
                        //如果获取到的值不为null则添加为xml的一个节点
                        if(null != childNode) {
                        	classNode.appendChild(childNode);
                        }
                	}
                }
                ele.appendChild(classNode);
                handlePropertyWhenGenerateEnd(doc, ele);
        	}

            //生成xml
            doc.appendChild(ele);
			doc.setXmlVersion("1.0");
			doc.setXmlStandalone(true);
			
			return XmlUtil.transDocToXml(doc, getEncoding());
        }

        return null;
    }
    
	/**
     * 获取属性值的toString()值
     * Function Description here
     * @param propertyValue 属性值
     * @param needEmpty 是否需要空属性
     * @return
     */
    @Override
    protected Node getPropertyValue(Document doc, String nodeName, Object propertyValue, boolean needEmpty, String[] includeMethods, String[] excludeMethods) 
    	throws Exception {
    	if (propertyValue instanceof List) {
    		//list类型的节点处理
    		List<?> list = (List<?>)propertyValue;
    		int size = list.size();

    		if(size > 0) {
    			Node rootNode = doc.createElement(nodeName);
	    		for(int i = 0; i < size; i++) {
	    			Node parentNode = doc.createElement(getSubListName());
	    			super.generateObjectNode(doc, list.get(i), parentNode, needEmpty, includeMethods, excludeMethods);
	    			
	    			//如果有子节点则添加进来
	    			if(parentNode.hasChildNodes()) {
	    				rootNode.appendChild(parentNode);
	    			}
	    		}
	    		
	    		return rootNode;
    		} else {
    			return null;
    		}
        } else {
        	return super.getPropertyValue(doc, nodeName, propertyValue, needEmpty, includeMethods, excludeMethods);
        }
    }
    
    public static void main(String[] args) {
    }
}







