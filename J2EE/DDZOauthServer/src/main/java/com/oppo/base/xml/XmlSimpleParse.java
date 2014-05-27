/**
 * XmlParse.java
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

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Node;

import com.oppo.base.common.OConstants;

/**
 * ClassName:XmlParse
 * Function: Xml文件信息读取
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-26  下午01:03:37
 */
public abstract class XmlSimpleParse {
	/**
	 * 根据文件路径读取
	 * @param filePath
	 */
	public void readFromFile(String filePath) {
		readFromFile(new File(filePath));
	}
	
	/**
	 * 从文件中读取
	 * @param filePath
	 */
	public void readFromFile(File xmlFile) {
		if(!xmlFile.exists()) {
			return;
		}
		
		parse(xmlFile);
	}
	
	/**
	 * 从输入流中读取
	 * @param inputStream 输入流
	 * @return
	 */
	public void readFromFile(InputStream inputStream) {
		parse(inputStream);
	}
	
	private void parse(Object xmlObj) {
		reset();
		if(null == xmlObj) {
			return;
		}

		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		try{
			DocumentBuilder db = dbf.newDocumentBuilder();
			Document doc = null;
			if(xmlObj instanceof String){
				doc = db.parse(new ByteArrayInputStream(xmlObj.toString().getBytes(OConstants.DEFAULT_ENCODING)));
			} else if(xmlObj instanceof File){
				doc = db.parse((File)xmlObj);
			} else if(xmlObj instanceof InputStream) {
				doc = db.parse((InputStream)xmlObj);
			}
			
			if(null != doc) {
				visitNode(doc.getDocumentElement());
			}
		}catch(Exception ex) {
			reset();
			
//			ex.printStackTrace();
		}
	}

	/**
	 * 遍历节点
	 * @param node
	 */
	protected abstract void visitNode(Node node) throws Exception;
	
	/**
	 * 清除已解析到的节点信息
	 */
	public abstract void reset();
}

