package com.oppo.base.xml;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import com.oppo.base.common.StringUtil;

/**
 * XML配置文件读取类
 * 只解析最简单的node-text结构，如<Cacheable>true</Cacheable>
 * 解析完成后将节点名作为key，节点值作为value存入到Map中供获取
 * @author yangbo
 * 2011-1-7
 */
public class XmlConfigReader extends ConfigReader {
	/**
	 * 保存所有配置文件,避免多次读取
	 */
	private static HashMap<String, XmlConfigReader> xmlReaderTable = new HashMap<String, XmlConfigReader>();
	private static final ReentrantLock LOCK = new ReentrantLock();
	private boolean isChanged;	//配置文件是否发生更改

    protected HashMap<String, String> configMap; //配置存放的map

    private XmlConfigReader(File file, int reloadMode) {
    	super(file);
    	
    	this.isChanged = true;
        this.configMap = new HashMap<String, String>();
        this.setReloadMode(reloadMode);
        this.setInterval(5 * 60 * 1000);
    }
    
    /**
     * 获取指定XML配置信息实例
     * @param filePath 配置文件路径
     */
    public static XmlConfigReader getXmlConfig(String filePath) {
        return getXmlConfig(new File(filePath));
    }
    
    /**
     * 获取指定XML配置信息实例
     * @param file 配置文件
     */
    public static XmlConfigReader getXmlConfig(File file) {
    	return getXmlConfig(file, REALTIME_RELOAD);
    }
    
    /**
     * 获取指定XML配置信息实例
     * @param file 配置文件
     */
    public static XmlConfigReader getXmlConfig(File file, int reloadMode) {
    	String filePath = file.getAbsolutePath();
    	LOCK.lock();
    	try { //防止多次读取
	        if (xmlReaderTable.containsKey(filePath)) {
	            return xmlReaderTable.get(filePath);
	        } else {
	        	XmlConfigReader reader = new XmlConfigReader(file, reloadMode);
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
    public HashMap<String, String> getConfigMap() {
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
    	
    	configMap = xsp.getXmlMap();
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
	
	public static void main(String[] args) {
		File file = new File("G:\\test.xml");
		
		XmlConfigReader cr = XmlConfigReader.getXmlConfig(file);
		Map<String, String> configMap = cr.getConfigMap();
		StringBuilder sb = new StringBuilder();
		for(String key : configMap.keySet()) {
			sb.append(key + ":" + configMap.get(key) + "\t");
		}
			
		System.out.println(sb.toString());
	}
}
