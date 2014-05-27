/**
 * XmlObjectParser.java
 * com.oppo.base.xml
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-9 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.xml;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Date;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;
import org.xml.sax.InputSource;

import com.oppo.base.beanutils.BeanUtil;
import com.oppo.base.beanutils.DefaultNameTransfer;
import com.oppo.base.beanutils.INameTransfer;
import com.oppo.base.beanutils.IValueTransfer;
import com.oppo.base.beanutils.MethodsUtil;
import com.oppo.base.beanutils.StringValueTransfer;
import com.oppo.base.common.OConstants;
import com.oppo.base.time.TimeFormat;

/**
 * ClassName:XmlObjectParser
 * Function: 将一个独立的xml节点解析为对象的工具
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-8-9  下午03:38:38
 */
public class XmlObjectParser {
	private INameTransfer nameTransfer;
	private IValueTransfer<String> valueTransfer;
	private Element rootNode;
	private String encoding;

	private static INameTransfer DEFAULT_NAME_TRANSFER = new DefaultNameTransfer();
	private static IValueTransfer<String> DEFAULT_VALUE_TRANSFER = new StringValueTransfer();

	public XmlObjectParser() {
	}

	/**
	 * 根据指定的类型解析xml
	 * @param xmlText 包含数据的XML
	 * @param tClass 数据实体类型
	 */
    public <T> T getObjectFromXml(String xmlText, Class<T> tClass) throws Exception {
		return getObjectFromXmlObj(new ByteArrayInputStream(xmlText.getBytes(getEncoding())), tClass);
    }


    /**
	 * 根据指定的类型解析xml
	 * @param xmlText 包含数据的文件
	 * @param tClass 数据实体类型
	 */
    public <T> T getObjectFromXml(File xmlFile, Class<T> tClass) throws Exception {
    	return getObjectFromXmlObj(xmlFile, tClass);
    }

    /**
	 * 根据指定的类型解析xml
	 * @param inputStream 包含数据的流
	 * @param tClass 数据实体类型
	 */
    public <T> T getObjectFromXml(InputStream inputStream, Class<T> tClass) throws Exception {
    	return getObjectFromXmlObj(inputStream, tClass);
    }

    /**
	 * 根据指定的类型解析xml
	 * @param obj 包含数据的对象(文件或流)
	 * @param tClass 数据实体类型
	 */
    public <T> T getObjectFromXmlObj(Object obj, Class<T> tClass) throws Exception {
    	clearState();

    	DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();
		Document doc = null;

		if(obj instanceof InputStream) {
			doc = db.parse((InputStream)obj);
		} else if(obj instanceof InputSource) {
			doc = db.parse((InputSource)obj);
		} else if(obj instanceof File) {
			doc = db.parse((File)obj);
		} else {
			throw new Exception("unknown type of source");
		}

		Element rootNode = doc.getDocumentElement();
		if(checkRootNode(rootNode)) {
			this.rootNode = rootNode;
			return parseXmlNode(rootNode, tClass);
		} else {
			throw new Exception("root node error");
		}
    }

    /**
     * 将节点中的子节点解析为对象属性
     * @param <T> 实体类型
     * @param node Xml节点
     * @param tClass
     * @param nameTrans xml节点与实体属性名称转换器
     * @return
     * @throws Exception
     */
    public <T> T parseXmlNode(Node node, Class<T> tClass) throws Exception {
        T tObj = tClass.newInstance();//创建实体
        boolean isSetParam = false;

        NodeList nodeList = node.getChildNodes();
        int listLen = nodeList.getLength();
        for (int i = 0; i < listLen; i++) {
            Node subNode = nodeList.item(i);

            //解析子节点,如果子节点为文本则作为一个普通属性,否则为list
            NodeList subNodeList = subNode.getChildNodes();
            int subNodeCount = subNodeList.getLength();
            if (subNodeCount == 1 && (subNodeList.item(0) instanceof Text)) {
            	boolean isTextSet = handleTextNode(subNode, (Text)subNodeList.item(0), tObj);
                if(isTextSet) {
                	isSetParam = true;
                }
            } else if (subNodeCount >= 1) { //这种情况则当做List解析
            	boolean isNodeSet = handleNotTextNode(subNode, subNodeList, tObj);
            	if(isNodeSet) {
            		isSetParam = true;
            	}
            }
        }

        handleXmlTextWhenParseEnd();

        return isSetParam ? tObj : null;
    }

    /**
     * 处理text类型的子节点
     * @param
     * @return 该节点是否有效
     */
    protected <T> boolean handleTextNode(Node parentNode, Text textNode, T tObj) throws Exception {
    	String value = textNode.getWholeText().trim();
        if (!"".equals(value)) {//如果非空
        	String name = parentNode.getNodeName();
        	this.handleXmlTextWhenParse(name, value);

        	return BeanUtil.setValueToObject(getNameTransfer(), getValueTransfer(), tObj, MethodsUtil.getMethods(tObj.getClass()), name, value);
        }

        return false;
    }

    /**
     * 处理非text类型的子节点
     * @param
     * @return
     */
    protected <T> boolean handleNotTextNode(Node parentNode, NodeList subNodeList, T tObj) throws Exception {
    	return false;
    }

    /**
     * 检查根节点是否合法
     * @param node 根节点
     * @return
     */
    protected boolean checkRootNode(Node node) {
    	return null != node;
    }

    /**
     * 利用指定对象生成xml
     * @param obj 需要解析的对象
     * @param documentName 根节点
     * @return
     */
    public String generateObjectXML(Object obj, String documentName) throws Exception {
    	return generateObjectXML(obj, documentName, false, null, null);
    }

    /**
     * 利用指定对象生成xml
     * @param obj 需要解析的对象
     * @param documentName 根节点
     * @param needEmpty 是否加上属性为空的节点
     * @return
     */
    public String generateObjectXML(Object obj, String documentName, boolean needEmpty) throws Exception {
    	return generateObjectXML(obj, documentName, needEmpty, null, null);
    }

    /**
     * 利用指定对象生成xml
     * @param obj 需要解析的对象
     * @param documentName 根节点
     * @param includeMethods 只解析对应的字段，如果此属性存在值则忽略excludeMethod
     * @param excludeMethods 解析时去掉此属性中指定的字段
     * @return
     */
    public String generateObjectXML(Object obj, String documentName, String[] includeMethods, String[] excludeMethods) throws Exception {
    	return generateObjectXML(obj, documentName, false, includeMethods, excludeMethods);
    }

    /**
     * 利用指定对象生成xml
     * @param obj 需要解析的对象
     * @param documentName 根节点
     * @param needEmpty 是否加上属性为空的节点
     * @param includeMethods 只解析对应的字段，如果此属性存在值则忽略excludeMethod
     * @param excludeMethods 解析时去掉此属性中指定的字段
     * @return
     */
    public String generateObjectXML(Object obj, String documentName, boolean needEmpty, String[] includeMethods, String[] excludeMethods) throws Exception {
    	clearState();

    	if (obj != null) {
        	//生成document
        	DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        	DocumentBuilder db = dbf.newDocumentBuilder();
			Document doc = db.newDocument();
			Element ele = doc.createElement(documentName);

        	generateObjectNode(doc, obj, ele, needEmpty, includeMethods, excludeMethods);

            handlePropertyWhenGenerateEnd(doc, ele);

            //生成xml
            doc.appendChild(ele);
			doc.setXmlVersion("1.0");
			doc.setXmlStandalone(true);

			return XmlUtil.transDocToXml(doc, getEncoding());
        }

        return null;
    }

    /**
     * 利用指定对象生成xml
     * @param doc
     * @param obj 需要解析的对象
     * @param parentNode 父节点
     * @param needEmpty 是否加上属性为空的节点
     * @return
     */
    public void generateObjectNode(Document doc, Object obj, Node parentNode, boolean needEmpty) throws Exception {
    	generateObjectNode(doc, obj, parentNode, needEmpty, null, null);
    }

    /**
     * 利用指定对象生成xml
     * @param doc
     * @param obj 需要解析的对象
     * @param parentNode 父节点
     * @param needEmpty 是否加上属性为空的节点
     * @param includeMethods 只解析对应的字段，如果此属性存在值则忽略excludeMethod
     * @param excludeMethods 解析时去掉此属性中指定的字段
     * @return
     */
    public void generateObjectNode(Document doc, Object obj, Node parentNode, boolean needEmpty, String[] includeMethods, String[] excludeMethods) throws Exception {
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

        		//判断是否可以被解析
				if(!canParse(nodeName, includeMethods, excludeMethods)) {
					continue;
				}

        		Object value = method.invoke(obj);
                Node childNode = getPropertyValue(doc, nodeName, value, needEmpty, includeMethods, excludeMethods);
                //如果获取到的值不为null则添加为xml的一个节点
                if(null != childNode) {
                	parentNode.appendChild(childNode);
                }
        	}
        }
    }

    /**
     * 获取属性值的toString()值
     * Function Description here
     * @param propertyValue 属性值
     * @param needEmpty 是否需要空属性
     * @return
     */
    protected Node getPropertyValue(Document doc, String nodeName, Object propertyValue, boolean needEmpty, String[] includeMethods, String[] excludeMethods)
    	throws Exception {
    	if ((propertyValue instanceof Iterable) || (propertyValue instanceof Map)) {
            return null;
        }

    	String strValue = "";
    	if(null != propertyValue) {
    		//如果是Date类型
    		if(propertyValue instanceof Date) {
    			strValue = TimeFormat.getTimeString((Date)propertyValue);
    		} else {
    			strValue = propertyValue.toString().trim();
    		}
    	}

        if ((!needEmpty && !strValue.equals("")) || needEmpty) {
        	Element subEle = doc.createElement(nodeName);

        	subEle.appendChild(doc.createTextNode(strValue));

        	handlePropertyWhenGenerate(strValue);

            return subEle;
        }

        return null;
    }

    /**
	 * 判断指定名称在规则下是否可以进行解析
	 * @param
	 * @return
	 */
	protected boolean canParse(String name, String[] includeMethods, String[] excludeMethods) {
		if(null != includeMethods) {
			//优先包含的
			for(String includeMethod : includeMethods) {
				if(name.equals(includeMethod)) {
					return true;
				}
			}

			return false;
		} else if(null != excludeMethods) {
			for(String excludeMethod : excludeMethods) {
				if(name.equals(excludeMethod)) {
					return false;
				}
			}

			return true;
		}

		return true;
	}

    /**
     * 提供额外处理属性的方法,属性处理完成后对根节点的处理
     * @param rootNode
     * @return
     */
    protected void handlePropertyWhenGenerateEnd(Document doc, Node rootNode) {
    }

    /**
     * 提供额外处理属性的方法,在生成xml时若有特殊要求可覆盖此方法
     * 如利用节点进行签名等
     * @param propertyString 属性的toString值
     * @return
     */
    protected void handlePropertyWhenGenerate(String propertyString) {
    }

    /**
     * 提供额外处理xml text节点的方法,在解析xml时若有特殊要求可覆盖此方法
     * 如利用节点进行签名等
     * @param nodeName 节点名
     * @param xmlText xml text节点值
     * @return
     */
    protected void handleXmlTextWhenParse(String nodeName, String xmlText) {
    }

    /**
     * 提供额外处理xml text的方法,在解析xml完成时若有特殊要求可覆盖此方法
     * 如利用节点进行签名验证等
     * @param xmlText xml text节点值
     * @return
     */
    protected void handleXmlTextWhenParseEnd() {
    }

    protected void clearState() {
    }

	/**
	 * 获取名称转换器
	 * @return  the nameTransfer
	 * @since   Ver 1.0
	 */
	public INameTransfer getNameTransfer() {
		if(null == nameTransfer) {
        	nameTransfer = DEFAULT_NAME_TRANSFER;
        }

		return nameTransfer;
	}

	/**
	 * 设置名称转换器
	 * @param   nameTransfer
	 * @since   Ver 1.0
	 */
	public void setNameTransfer(INameTransfer nameTransfer) {
		this.nameTransfer = nameTransfer;
	}

	/**
	 * 获取值转换器
	 * @return  the valueTransfer
	 * @since   Ver 1.0
	 */
	public IValueTransfer<String> getValueTransfer() {
		if(null == valueTransfer) {
			valueTransfer = DEFAULT_VALUE_TRANSFER;
		}

		return valueTransfer;
	}

	/**
	 * 设置值转换器
	 * @param   valueTransfer
	 * @since   Ver 1.0
	 */
	public void setValueTransfer(IValueTransfer<String> valueTransfer) {
		this.valueTransfer = valueTransfer;
	}

	/**
	 * 获取encoding
	 * @return  the encoding
	 * @since   Ver 1.0
	 */
	public String getEncoding() {
		if(null == encoding) {
			encoding = OConstants.DEFAULT_ENCODING;
		}

		return encoding;
	}

	/**
	 * 设置encoding
	 * @param   encoding
	 * @since   Ver 1.0
	 */
	public void setEncoding(String encoding) {
		this.encoding = encoding;
	}

	/**
	 * 获取根节点
	 * @param
	 * @return
	 */
	public Element getRootNode() {
		return rootNode;
	}
}

