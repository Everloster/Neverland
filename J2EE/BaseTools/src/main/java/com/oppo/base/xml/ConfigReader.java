/**
 * ConfigReader.java
 * com.oppo.base.xml
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-2-20 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.oppo.base.xml;

import java.io.File;

/**
 * ClassName:ConfigReader <br>
 * Function: 配置文件读取 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-2-20  下午04:54:51
 */
public class ConfigReader {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	 /**
     * 实时判断文件是否有改动
     */
    public static final int REALTIME_RELOAD = 0;
    
    /**
     * 每隔一段时间判断是否有改变
     */
    public static final int INTERVEL_RELOAD = 1;
    
    /**
     * 只加载一次
     */
    public static final int ONETIME_RELOAD = 2;
    
    private File file; 				//配置文件所在路径
    private long lastModifyTime = 0; 	//配置文件最后一次的修改时间
    private long lastLoadTime = 0;		//最后一次读取配置文件的时间 
    private int reloadMode;				//重新读取文件的模式
    private long interval;				//重新读取文件的时间间隔，间隔读取的方式有效
    
    public ConfigReader() {
    }
    
    public ConfigReader(String filePath) {
    	this(new File(filePath));
    }
    
    public ConfigReader(File file) {
    	this.file = file;
    }
    
    /**
     * 是否需要重新加载数据
     * @param 
     * @return
     */
    protected boolean needReload() {
    	switch(reloadMode) {
    	case REALTIME_RELOAD:
    		return needReloadRealtime();
    	case INTERVEL_RELOAD:
    		return needReloadInterval();
    	case ONETIME_RELOAD:
    		return needReloadOnetime();
    	default:
    		return needReloadRealtime();
    	}
    }
    
    //实时数据加载判断
    protected boolean needReloadRealtime() {
    	if (!file.exists()) {
    		return true;
    	}
    	
    	long fileModifyTime = file.lastModified();
        if (fileModifyTime == lastModifyTime) {
            return false;
        }
        
        lastModifyTime = fileModifyTime;
        
        return true;
    }
    
    //实时数据加载判断
    protected boolean needReloadInterval() {
    	long now = System.currentTimeMillis();
    	if(now - lastLoadTime > interval) {
    		lastLoadTime = now;
    		return needReloadRealtime();
    	} else {
    		return false;
    	}
    }
    
    //只读一次
    protected boolean needReloadOnetime() {
    	if(lastLoadTime <= 0) {
    		lastLoadTime = System.currentTimeMillis();
    		return true;
    	} else {
    		return false;
    	}
    }
    
	/**
	 * 获取file
	 * @return  the file
	 * @since   Ver 1.0
	 */
	public File getFile() {
		return file;
	}
	/**
	 * 设置file
	 * @param   file    
	 * @since   Ver 1.0
	 */
	public void setFile(File file) {
		this.file = file;
	}
	/**
	 * 获取lastModifyTime
	 * @return  the lastModifyTime
	 * @since   Ver 1.0
	 */
	public long getLastModifyTime() {
		return lastModifyTime;
	}
	/**
	 * 获取lastLoadTime
	 * @return  the lastLoadTime
	 * @since   Ver 1.0
	 */
	public long getLastLoadTime() {
		return lastLoadTime;
	}

	/**
	 * 获取reloadMode
	 * @return  the reloadMode
	 * @since   Ver 1.0
	 */
	public int getReloadMode() {
		return reloadMode;
	}
	/**
	 * 设置reloadMode
	 * @param   reloadMode    
	 * @since   Ver 1.0
	 */
	public void setReloadMode(int reloadMode) {
		this.reloadMode = reloadMode;
	}
	/**
	 * 获取interval
	 * @return  the interval
	 * @since   Ver 1.0
	 */
	public long getInterval() {
		return interval;
	}
	/**
	 * 设置interval
	 * @param   interval    
	 * @since   Ver 1.0
	 */
	public void setInterval(long interval) {
		this.interval = interval;
	}
}

