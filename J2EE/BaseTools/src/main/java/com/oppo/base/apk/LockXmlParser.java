/**
 * ManifestXmlParser.java
 * com.oppo.base.apk
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-16 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.apk;

import com.oppo.base.apk.entity.LockProperty;

/**
 * ClassName:LockXmlParser
 * Function: res/xml/lock.xml 文件解析
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-16  上午09:56:31
 * 
 */
public class LockXmlParser extends ApkXmlParser {
	private static final String LOCK_TAG_NAME = "apklock";
	private static final String LOCK_NODE_NAME = "lockName";
	private static final String LOCK_NODE_AUTHOR = "lockAuthor";
	private static final String LOCK_NODE_VERSION = "lockVersion";
	private static final String LOCK_NODE_RESOLUTION = "lockResolution";
	private static final String LOCK_NODE_DESC = "lockDescription";
	private static final String LOCK_NODE_THUMBNAIL = "lockThumbnail";
	private static final String LOCK_NODE_THUMBLEFT = "lockThumbleft";
	private static final String LOCK_NODE_THUMBRIGHT = "lockThumbright";
	
	private LockProperty lockProperty;
	private String tagName;
	
	public LockXmlParser() {
		super();
	}
	
	@Override
	public void reset() {
		super.reset();
		
		lockProperty = new LockProperty();
	}
	
	public LockProperty getLockProperty() {
		return lockProperty;
	}
	
	@Override
	protected void appendStartTag(String format, String indent, String tagName) {
		super.appendStartTag(format, indent, tagName);
		this.tagName = tagName;
	}
	
	@Override
	protected void appendEndTag(String format, String indent, String namespace, String name) {
		super.appendEndTag(format, indent, namespace, name);
		
		this.tagName = null;
	}
	
	@Override
	protected void appendNode(String format, String indent, String namespace, String nodeName, String value) {
		super.appendNode(format, indent, namespace, nodeName, value);
		
		if(LOCK_TAG_NAME.equals(tagName)) {
			if(LOCK_NODE_NAME.equals(nodeName)) {
				lockProperty.setName(value);
			} else if(LOCK_NODE_AUTHOR.equals(nodeName)){
				lockProperty.setAuthor(value);
			} else if(LOCK_NODE_VERSION.equals(nodeName)){
				lockProperty.setVersion(value);
			} else if(LOCK_NODE_RESOLUTION.equals(nodeName)){
				lockProperty.setResolution(value);
			} else if(LOCK_NODE_DESC.equals(nodeName)){
				lockProperty.setDescription(value);
			} else if(LOCK_NODE_THUMBNAIL.equals(nodeName)){
				lockProperty.setThumbnail(value);
			} else if(LOCK_NODE_THUMBLEFT.equals(nodeName)){
				lockProperty.setThumbLeft(value);
			} else if(LOCK_NODE_THUMBRIGHT.equals(nodeName)){
				lockProperty.setThumbRight(value);
			}
		}
		
	}
	
	public static void main(String[] args) {
		
	}
}

