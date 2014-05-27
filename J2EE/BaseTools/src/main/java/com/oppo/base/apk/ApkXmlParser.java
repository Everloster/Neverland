/**
 * ApkParser.java
 * com.oppo.base.apk
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-15 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 */

package com.oppo.base.apk;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import android.util.TypedValue;
import brut.androlib.res.decoder.AXmlResourceParser;

import com.oppo.base.common.OConstants;

/**
 * ClassName:ApkParser
 * Function: apk包中的xml解析
 *
 * 如果对特殊的属性解析，请重写appendNode方法
 *
 *
 * 调用方式：
 * 	ApkXmlParser parser = new ApkXmlParser();
 *  parser.parse("/var/resource/xml/help.xml");
 *  String xmlString = parser.getXmlString();
 * xmlString即/var/resource/xml/help.xml解析后的xml字符串
 * @author 80036381
 * @version
 * @since Ver 1.1
 * @Date 2011-7-15 下午04:07:08
 *
 */
public class ApkXmlParser {
	String NO_NAMESPACE = "";

    // ----------------------------------------------------------------------------
    // EVENT TYPES as reported by next()

    /**
     *
     */
	private static final int START_DOCUMENT = 0;

    /**
     *
     */
	private static final int END_DOCUMENT = 1;

    /**
     *
     */
    private static final int START_TAG = 2;

    /**
     *
     */
    private static final int END_TAG = 3;


    /**
     *
     */
    private static final int TEXT = 4;

    // ----------------------------------------------------------------------------
    // additional events exposed by lower level nextToken()

//    /**
//     *
//     */
//    private static final int CDSECT = 5;
//
//    /**
//     *
//     */
//    private static final int ENTITY_REF = 6;
//
//    /**
//     *
//     */
//    private static final int IGNORABLE_WHITESPACE = 7;
//
//    /**
//     *
//     */
//    private static final int PROCESSING_INSTRUCTION = 8;
//
//    /**
//     *
//     */
//    private static final int COMMENT = 9;
//
//    /**
//     *
//     */
//    private static final int DOCDECL = 10;
//
//    /**
//     *
//     */
//    private static final String [] TYPES = {
//        "START_DOCUMENT",
//            "END_DOCUMENT",
//            "START_TAG",
//            "END_TAG",
//            "TEXT",
//            "CDSECT",
//            "ENTITY_REF",
//            "IGNORABLE_WHITESPACE",
//            "PROCESSING_INSTRUCTION",
//            "COMMENT",
//            "DOCDECL"
//    };

	private static final String SPACE = " ";
	private static final String INDENT = " ";
	private static final int INDENT_LENGTH = INDENT.length();

	private StringBuilder xmlBuilder; // xml文件内容
	private Map<String, String> nodeMap;// xml节点map
	private String encoding; // xml文件编码

	public ApkXmlParser() {
		encoding = OConstants.DEFAULT_ENCODING;
	}

	/**
	 * 解析xml资源
	 *
	 * @param
	 * @return
	 */
	public void parse(String filePath) throws Exception {
		parse(new FileInputStream(filePath));
	}

	/**
	 * 解析xml资源
	 *
	 * @param
	 * @return
	 */
	public void parse(File file) throws Exception {
		parse(new FileInputStream(file));
	}

	/**
	 * 解析xml资源
	 *
	 * @param
	 * @return
	 */
	public void parse(InputStream inputStream) throws Exception {
		reset();
		if (null == encoding) {
			encoding = OConstants.DEFAULT_ENCODING;
		}

		AXmlResourceParser parser = new AXmlResourceParser();
		parser.open(inputStream);
		StringBuilder indent = new StringBuilder(10);

		while (true) {
			int type = parser.next();

			if (type == END_DOCUMENT) {
				break;
			}

			switch (type) {
			case START_DOCUMENT:	//文档开头
				appendStartDoc(String.format(
						"<?xml version=\"1.0\" encoding=\"%s\"?>\n", encoding));
				break;

			case START_TAG:		//标签开头
				// 获取tag name
				String tagName = getNamespacePrefix(parser.getNamespace())
						+ parser.getName();

				appendStartTag("%s<%s", indent.toString(), tagName);
				indent.append(INDENT);

				int namespaceCountBefore = parser.getNamespaceCount(parser.getDepth() - 1);
				int namespaceCount = parser.getNamespaceCount(parser.getDepth());
				for (int i = namespaceCountBefore; i != namespaceCount; ++i) {
					appendFormat("%sxmlns:%s=\"%s\"", SPACE,
							parser.getNamespacePrefix(i),
							parser.getNamespaceUri(i));
				}

				//标签中的属性
				for (int i = 0; i != parser.getAttributeCount(); ++i) {
					appendNode("%s%s%s=\"%s\"", SPACE, getNamespacePrefix(parser.getAttributePrefix(i)),
							parser.getAttributeName(i), getAttributeValue(parser, i));
				}
				append(">\n");
				break;

			case END_TAG:		//标签结束
				indent.setLength(indent.length() - INDENT_LENGTH);
				appendEndTag("%s</%s%s>\n", indent.toString(),
						getNamespacePrefix(parser.getNamespace()),
						parser.getName());
				break;

			case TEXT:			//TEXT内容
				appendText("%s%s", indent, parser.getText());
				break;

			}
		}
	}

	/**
	 * 清空解析到的xml内容 Function Description here
	 *
	 * @param
	 * @return
	 */
	public void reset() {
		if (null == xmlBuilder) {
			xmlBuilder = new StringBuilder();
		} else {
			xmlBuilder.delete(0, xmlBuilder.length());
		}

		if (null == nodeMap) {
			nodeMap = new HashMap<String, String>();
		} else {
			nodeMap.clear();
		}
	}

	/**
	 * 获取xml内容
	 *
	 * @param
	 * @return
	 */
	public String getXmlString() {
		return xmlBuilder.toString();
	}

	/**
	 * 获取xml编码
	 *
	 * @return the encoding
	 * @since Ver 1.0
	 */
	public String getEncoding() {
		return encoding;
	}

	/**
	 * 设置xml编码,如果设置为null,将使用默认编码utf-8
	 *
	 * @param encoding
	 * @since Ver 1.0
	 */
	public void setEncoding(String encoding) {
		this.encoding = encoding;
	}

	protected void append(String line) {
		xmlBuilder.append(line);
	}

	/**
	 * 添加xml的声明
	 *
	 * @param
	 * @return
	 */
	protected void appendStartDoc(String doc) {
		append(doc);
	}

	protected void appendFormat(String format, Object... args) {
		append(String.format(format, args));
	}

	/**
	 * 添加xml的起始标签
	 *
	 * @param
	 * @return
	 */
	protected void appendStartTag(String format, String indent, String tagName) {
		append(String.format(format, indent, tagName));
	}

	/**
	 * 添加xml的结束标签
	 *
	 * @param
	 * @return
	 */
	protected void appendEndTag(String format, String indent, String namespace,
			String name) {
		append(String.format(format, indent, namespace, name));
	}

	/**
	 * 添加xml的Text
	 *
	 * @param
	 * @return
	 */
	protected void appendText(String format, StringBuilder indent,
			String namespace) {
		append(String.format(format, indent, namespace));
	}

	/**
	 * 添加xml的node
	 *
	 * @param
	 * @return
	 */
	protected void appendNode(String format, String indent, String namespace, String nodeName,
			String value) {
		append(String.format(format, indent, namespace, nodeName, value));

		// 添加到节点map中
		nodeMap.put(nodeName, value);
	}

	private String getNamespacePrefix(String prefix) {
		if (prefix == null || prefix.length() == 0) {
			return "";
		}
		return prefix + ":";
	}

	private String getAttributeValue(AXmlResourceParser parser, int index) {
		int type = parser.getAttributeValueType(index);
		int data = parser.getAttributeValueData(index);
		if (type == TypedValue.TYPE_STRING) {
			return parser.getAttributeValue(index);
		}
		if (type == TypedValue.TYPE_ATTRIBUTE) {
			return String.format("?%s%08x", getPackage(data), data);
		}
		if (type == TypedValue.TYPE_REFERENCE) {
			return String.format("@%s%08x", getPackage(data), data);
		}
		if (type == TypedValue.TYPE_FLOAT) {
			return String.valueOf(Float.intBitsToFloat(data));
		}
		if (type == TypedValue.TYPE_INT_HEX) {
			return String.format("0x%08x", data);
		}
		if (type == TypedValue.TYPE_INT_BOOLEAN) {
			return data != 0 ? "true" : "false";
		}
		if (type == TypedValue.TYPE_DIMENSION) {
			return Float.toString(complexToFloat(data))
					+ DIMENSION_UNITS[data & TypedValue.COMPLEX_UNIT_MASK];
		}
		if (type == TypedValue.TYPE_FRACTION) {
			return Float.toString(complexToFloat(data))
					+ FRACTION_UNITS[data & TypedValue.COMPLEX_UNIT_MASK];
		}
		if (type >= TypedValue.TYPE_FIRST_COLOR_INT
				&& type <= TypedValue.TYPE_LAST_COLOR_INT) {
			return String.format("#%08x", data);
		}
		if (type >= TypedValue.TYPE_FIRST_INT
				&& type <= TypedValue.TYPE_LAST_INT) {
			return String.valueOf(data);
		}
		return String.format("<0x%x, type 0x%02x>", data, type);
	}

	private String getPackage(int id) {
		if (id >>> 24 == 1) {
			return "android:";
		}
		return "";
	}

	// ///////////////////////////////// ILLEGAL STUFF, DONT LOOK :)

	private float complexToFloat(int complex) {
		return (float) (complex & 0xFFFFFF00) * RADIX_MULTS[(complex >> 4) & 3];
	}

	private static final float RADIX_MULTS[] = { 0.00390625F, 3.051758E-005F,
			1.192093E-007F, 4.656613E-010F };
	private static final String DIMENSION_UNITS[] = { "px", "dip", "sp", "pt",
			"in", "mm", "", "" };
	private static final String FRACTION_UNITS[] = { "%", "%p", "", "", "", "",
			"", "" };

	public static void main(String[] arguments) throws Exception {
		ApkXmlParser parser = new ApkXmlParser();
		parser.parse("h:\\lock.xml");
		String xmlString = parser.getXmlString();
		System.out.println(xmlString);
	}
}
