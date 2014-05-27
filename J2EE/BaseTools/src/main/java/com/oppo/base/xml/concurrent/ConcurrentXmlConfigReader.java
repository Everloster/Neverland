/**
 * ConcurrentXmlConfigReader.java
 * com.oppo.base.xml.concurrent
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-29 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.oppo.base.xml.concurrent;

import java.io.File;
import java.util.concurrent.ConcurrentHashMap;

import com.oppo.base.common.StringUtil;
import com.oppo.base.xml.ConfigReader;
import com.oppo.base.xml.XmlSimpleMapParse;

/**
 * ClassName:ConcurrentXmlConfigReader <br>
 * Function: XML配置文件读取类，
 * 只解析最简单的node-text结构，如<Cacheable>true</Cacheable>，
 * 解析完成后将节点名作为key，节点值作为value存入到Map中供获取 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-29  上午10:37:12
 */
public class ConcurrentXmlConfigReader extends ConfigReader {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * 保存所有配置文件,避免多次读取
	 */
	private static ConcurrentHashMap<String, ConcurrentXmlConfigReader> xmlReaderMap = new ConcurrentHashMap<String, ConcurrentXmlConfigReader>();
	private boolean isChanged;	//配置文件是否发生更改
	
    protected ConcurrentHashMap<String, String> configMap; //配置存放的map
    
    private ConcurrentXmlConfigReader(File file, int reloadMode) {
    	super(file);
    	this.isChanged = true;
        this.configMap = new ConcurrentHashMap<String, String>();
        this.setReloadMode(reloadMode);
        this.setInterval(5 * 60 * 1000);
    }
    
    /**
     * 获取指定XML配置信息实例
     * @param filePath 配置文件路径
     */
    public static ConcurrentXmlConfigReader getXmlConfig(String filePath) {
        return getXmlConfig(new File(filePath));
    }
    
    /**
     * 获取指定XML配置信息实例
     * @param file 配置文件
     */
    public static ConcurrentXmlConfigReader getXmlConfig(File file) {
    	return getXmlConfig(file, REALTIME_RELOAD);
    }
    
    /**
     * 获取指定XML配置信息实例
     * @param file 配置文件
     * @param reloadMode 文件信息重新加载的模式
     */
    public static ConcurrentXmlConfigReader getXmlConfig(File file, int reloadMode) {
    	String filePath = file.getAbsolutePath();
    	
    	//防止多次读取
    	ConcurrentXmlConfigReader reader = xmlReaderMap.get(filePath);
    	if(null == reader) {
    		reader = new ConcurrentXmlConfigReader(file, reloadMode);
    		ConcurrentXmlConfigReader oldReader = xmlReaderMap.putIfAbsent(filePath, reader);
    		if(null != oldReader) {
    			reader = oldReader;
    		}
    	}
	     
	    return reader;
    }
    
    /**
     * 获取所有配置信息
     * @return
     */
    public ConcurrentHashMap<String, String> getConfigMap() {
    	initialConfig();
    	return configMap;
    }
    
    /**
     * 获取配置内容
     * @param key
     */
    public String getConfig(String key) {
        if (StringUtil.isNullOrEmpty(key)) {
            return null;
        }

        initialConfig();
        if (null == configMap) {
            return null;
        }

        return configMap.get(key);
    }
    
    /**
     * 初始化配置,如果最后修改时间与最后读取配置文件的时间不一致,则重新读取
     */
    public void initialConfig() {
        if (!needReload()) {
        	isChanged = false;
            return;
        }
        
        reset();
        readConfig();
    }
    
    /**
     * 读取配置文件内容
     */
    protected void readConfig() {
    	XmlSimpleMapParse xsp = new XmlSimpleMapParse();
    	xsp.readFromFile(this.getFile());
    	
    	configMap = new ConcurrentHashMap<String, String>(xsp.getXmlMap());
    }
    
    /**
     * 重置数据
     */
    public void reset() {
    	isChanged = true;
    	
    	if(null != configMap) {
    		configMap.clear();
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

