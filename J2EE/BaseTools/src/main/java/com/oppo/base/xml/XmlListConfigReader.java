/**
 * XmlListConfigReader.java
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

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * ClassName:XmlListConfigReader
 * Function: XML配置文件读取类，可将xml中根节点下的第一级节点解析为Object
 * 例如存在文件D:\item_config.xml:
 * <Items>
 * 	<Item><value>item1</value></Item>
 * 	<Item><value>item2</value></Item>
 * </Items>
 * 则需要定义包含getValue、setValue方法的类Item其中类名不强制要求一致，方法名必须一致
 * 则解析方式为：List<Item> itemList = XmlListConfigReader.getXmlConfig("D:\\item_config.xml", Item.class).getConfigList();
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-26  下午12:52:30
 */
public class XmlListConfigReader<T> extends ConfigReader {
	/**
	 * 保存所有配置文件,避免多次读取
	 */
	private static HashMap<String, XmlListConfigReader> xmlReaderTable = new HashMap<String, XmlListConfigReader>();
	private static final ReentrantLock LOCK = new ReentrantLock();
	private boolean isChanged;	//配置文件是否发生更改
	
	private Class<T> cls;	//配置文件中的类的class
	private List<T> configList; //配置存放的map

    protected XmlListConfigReader(File file, Class<T> cls, int reloadMode) {
    	super(file);
    	this.isChanged = true;
        this.cls = cls;
        this.configList = new ArrayList<T>();
        this.setReloadMode(reloadMode);
        this.setInterval(5 * 60 * 1000);
    }
    
    /**
     * 获取指定XML配置信息实例
     * @param filePath 配置文件路径
     */
    public static <T> XmlListConfigReader<T> getXmlConfig(String filePath, Class<T> cls) {
        return getXmlConfig(new File(filePath), cls);
    }
    
    /**
     * 获取指定XML配置信息实例
     * @param file 配置文件
     */
    @SuppressWarnings("unchecked")
	public static <T> XmlListConfigReader<T> getXmlConfig(File file, Class<T> cls) {
    	return getXmlConfig(file, cls, REALTIME_RELOAD);
    }
    
    /**
     * 获取指定XML配置信息实例
     * @param file 配置文件
     */
    @SuppressWarnings("unchecked")
	public static <T> XmlListConfigReader<T> getXmlConfig(File file, Class<T> cls, int reloadMode) {
    	String filePath = file.getAbsolutePath();
    	LOCK.lock();
    	try { //防止多次读取
	        if (xmlReaderTable.containsKey(filePath)) {
	            return xmlReaderTable.get(filePath);
	        } else {
	        	XmlListConfigReader<T> reader = new XmlListConfigReader<T>(file, cls, reloadMode);
	            xmlReaderTable.put(filePath, reader);
	            return reader;
	        }
    	} finally {
    		LOCK.unlock();
    	}
    }
    
    /**
     * 获取所有配置信息
     * @return
     */
    public List<T> getConfigList() {
    	initialConfig();
    	return configList;
    }
    
    /**
     * 初始化配置,如果最后修改时间与最后读取配置文件的时间不一致,则重新读取
     */
    public void initialConfig() {
    	if (!needReload()) {
            return;
        }
        
        reset();
        readConfig();
    }
    
    /**
     * 读取配置文件内容
     */
    protected void readConfig() {
    	XmlSimpleListParse<T> xselp = new XmlSimpleListParse<T>();
    	xselp.setEntityClass(cls);
    	xselp.readFromFile(this.getFile());
    	
    	configList = xselp.getObjectList();
    }
    
    /**
     * 重置数据
     */
    public void reset() {
    	isChanged = true;
    	
    	if(null != configList) {
    		configList.clear();
    	}
    }

	/**
	 * 获取文件是否发生更改
	 * @return  如果更改返回true,否则返回false
	 * @since   Ver 1.0
	 */
	public boolean isChanged() {
		return isChanged;
	} 
}

