package com.oppo.base.xml;

import java.io.ByteArrayInputStream;
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

import com.oppo.base.beanutils.BeanUtil;
import com.oppo.base.beanutils.DefaultNameTransfer;
import com.oppo.base.beanutils.INameTransfer;
import com.oppo.base.beanutils.IValueTransfer;
import com.oppo.base.beanutils.StringValueTransfer;
import com.oppo.base.common.OConstants;
import com.oppo.base.time.TimeFormat;

/**
 * 解析xml为实体/将实体转换为xml的工具类
 * @author 80036381
 * 2011-1-7
 * 
 * @deprecated 由XmlObjectParser代替
 * @see com.oppo.base.xml.XmlObjectParser
 */
public class XmlEntityParser {
	private INameTransfer nameTransfer;
	private IValueTransfer<String> valueTransfer;
	private String encoding;
	private Method[] methods;
	
	public XmlEntityParser() {
		nameTransfer = new DefaultNameTransfer();
		valueTransfer = new StringValueTransfer();
		encoding = OConstants.DEFAULT_ENCODING;
	}
	
	/**
	 * 根据指定的类型解析xml
	 * @param xmlText 包含数据的XML
	 * @param tClass 数据实体类型
	 */
    public <T> T getObjectFromXml(String xmlText, Class<T> tClass) throws Exception {
    	return getObjectFromXml(xmlText, tClass, nameTransfer);
    }
    
    /**
	 * 根据指定的类型解析xml
	 * @param xmlText 包含数据的XML
	 * @param tClass 数据实体类型
	 * @param nameTrans xml节点与实体属性名称转换器
	 */
    public <T> T getObjectFromXml(String xmlText, Class<T> tClass, INameTransfer nameTrans) throws Exception {
    	clearState();
    	
    	DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();
		Document doc = db.parse(new ByteArrayInputStream(xmlText.getBytes(getEncoding())));
		Node node = doc.getDocumentElement();
		if(checkRootNode(node)) {
			return parseXmlNode(doc.getDocumentElement(), tClass, nameTrans);
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
    public <T> T parseXmlNode(Node node, Class<T> tClass, INameTransfer nameTrans) throws Exception {
        T tObj = tClass.newInstance();//创建实体
        boolean isSetParam = false;
     
        //如果未提供，则使用默认
        if(null == nameTrans) {
        	nameTrans = new DefaultNameTransfer();
        }
        
        NodeList nodeList = node.getChildNodes();
        int listLen = nodeList.getLength();
        for (int i = 0; i < listLen; i++) {
            Node subNode = nodeList.item(i);

            //解析子节点,如果子节点为文本则作为一个普通属性,否则为list
            NodeList subNodeList = subNode.getChildNodes();
            int subNodeCount = subNodeList.getLength();
            if (subNodeCount == 1 && (subNodeList.item(0) instanceof Text))
            {
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
    	Class<?> tClass = tObj.getClass();
    	methods = tClass.getMethods();
    	
    	String value = textNode.getWholeText().trim();
        if (!"".equals(value))//如果非空
        {
        	String name = parentNode.getNodeName();
        	this.handleXmlTextWhenParse(name, value);
        	
        	return BeanUtil.setValueToObject(nameTransfer, valueTransfer, tObj, methods, name, value);
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
     * @param needEmpty 是否加上属性为空的节点
     * @param xmlParse 负责生成xml的解析类
     * @return
     */
    public String generateObjectXML(Object obj, String documentName) throws Exception {
    	return generateObjectXML(obj, documentName, false);
    }
    
    /**
     * 利用指定对象生成xml
     * @param obj 需要解析的对象
     * @param documentName 根节点
     * @param needEmpty 是否加上属性为空的节点
     * @return
     */
    public String generateObjectXML(Object obj, String documentName, boolean needEmpty) throws Exception {
    	clearState();
    	
    	if (obj != null) {
        	//生成document
        	DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        	DocumentBuilder db = dbf.newDocumentBuilder();
			Document doc = db.newDocument();
			Element ele = doc.createElement(documentName);
			
        	
        	//逐个解析obj中的方法
        	Method[] ms = obj.getClass().getMethods();
        	int mLen = ms.length;
            for (int i = 0; i < mLen; i++) {
            	Method method = ms[i];
            	if(!Modifier.isPublic(method.getModifiers())) {
            		continue;
            	}
            	
            	String methodName = method.getName();
            	if(methodName.startsWith("get") && !methodName.equals("getClass")) {
            		String nodeName = nameTransfer.transferToNodeName(methodName);
            		Object value = method.invoke(obj);
                    String propertyValue = getPropertyValue(value, needEmpty);
                    //如果获取到的值不为null则添加为xml的一个节点
                    if(null != propertyValue) {
                    	Element subEle = doc.createElement(nodeName);
                    	subEle.appendChild(doc.createTextNode(propertyValue));
                    	ele.appendChild(subEle);
                    	
                    	handlePropertyWhenGenerate(propertyValue);
                    }
            	}
            }
            
            handlePropertyWhenGenerateEnd(doc, ele);
            
            //生成xml
            doc.appendChild(ele);
			doc.setXmlVersion("1.0");
			doc.setXmlStandalone(true);
			
			return XmlUtil.transDocToXml(doc, getEncoding());
        }

        return null;
    }
    
    protected Method getMethod(String methodName, Class<?>... parameterClass) {
    	if(null == parameterClass) {
    		parameterClass = new Class<?>[0];
    	}
    	
    	int pcLength = parameterClass.length;
    	int length = methods.length;
    	for(int i = 0; i < length; i++) {
    		Method method = methods[i];
    		//查找到同名方法
    		if(method.getName().equals(methodName)) {
    			//对比参数
    			Class<?>[] pClasses = method.getParameterTypes();
    			if(pClasses.length == pcLength) {
    				int j = 0;
    				for(; j < pcLength; j++) {
    					if(!pClasses[j].equals(parameterClass[j])) {
    						break;
    					}
    				}
    				
    				if(j == pcLength) {
    					return method;
    				}
    			}
    		}
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
    protected String getPropertyValue(Object propertyValue, boolean needEmpty) {
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
            return strValue;
        }
        
        return null;
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
}