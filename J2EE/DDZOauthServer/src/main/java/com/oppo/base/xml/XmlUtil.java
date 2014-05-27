/**
 * XmlUtil.java
 * com.oppo.base.xml
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-4 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.xml;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;
import org.w3c.dom.Element;

import com.oppo.base.common.OConstants;
import com.oppo.base.file.FileOperate;

/**
 * ClassName:XmlUtil
 * Function: xml文档转换工具类
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-4  下午05:11:31
 */
public class XmlUtil {
	/**
	 * 将Document对象转换为XML String
	 * @param doc
	 * @return
	 * @throws Exception
	 */
	public static String transDocToXml(Document doc) throws Exception{
		return transDocToXml(doc, OConstants.DEFAULT_ENCODING);
	}
	
	/**
	 * 将Document对象转换为XML String
	 * @param doc
	 * @param encoding 生成的xml的编码
	 * @return
	 * @throws Exception
	 */
	public static String transDocToXml(Document doc, String encoding) throws Exception{
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		
		transDocToOutput(doc, encoding, baos);
		
		return baos.toString(encoding);
	}
	
	/**
	 * 将Document对象存入到OutputStream中
	 * @param doc
	 * @param encoding 生成的xml的编码
	 * @param outputStream 接收输入的OutputStream
	 * @return
	 */
	public static void transDocToOutput(Document doc, String encoding, OutputStream outputStream) throws Exception {
		if(null == encoding) {
			encoding = OConstants.DEFAULT_ENCODING;
		}
		
		Transformer transformer=TransformerFactory.newInstance().newTransformer();
		DOMSource  domSource=new DOMSource(doc);
		
		StreamResult xmlResult=new StreamResult(outputStream);
		transformer.setParameter("encoding", encoding);
		transformer.transform(domSource, xmlResult);
	}
	
	/**
	 * 将map中的数据以xml的格式保存到指定的OutputStream中
	 * @param map 
	 * @param outputStream	保存数据的OutputStream
	 * @param documentName xml根节点名
	 * @param forceToSave 是否强制保存数据，如果未true，当map为空时会用空文件替换原文件，
	 * 			否则不处理
	 * @return
	 */
	public static void transMapToOutput(Map map, OutputStream outputStream, String documentName, boolean forceToSave) throws Exception {
		if(null == map) {
			if(forceToSave) {
				map = new HashMap(0);
			} else {
				//空数据不保存
				return;
			}
		}
		
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
    	DocumentBuilder db = dbf.newDocumentBuilder();
		Document doc = db.newDocument();
		Element ele = doc.createElement(documentName);
		
		//为每一个key-value生成节点
		for (Object key : map.keySet()) {
			if(null == key) {
				continue;
			}
			
			Element subEle = doc.createElement(key.toString());
			//对值进行处理
			Object value = map.get(key);
        	subEle.appendChild(doc.createTextNode((null == value) ? 
        			OConstants.EMPTY_STRING : value.toString()));
        	ele.appendChild(subEle);
		}
		
		//生成xml
        doc.appendChild(ele);
		doc.setXmlVersion("1.0");
		doc.setXmlStandalone(true);
		
		//将document输出到OutputStream中
		transDocToOutput(doc, null, outputStream);
	}
	
	public static void main(String[] args) {
		Map<String, String> map = new HashMap<String, String>();
		map.put("x1", "213");
		map.put("x2", "223");
		map.put("x3", "233");
		map.put("x4", "243");
		
		FileOutputStream fos = null;
		try {
			fos = new FileOutputStream("G:\\1.xml");
			transMapToOutput(map, fos, "test", true);
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			FileOperate.close(fos);
		}
	}
}

