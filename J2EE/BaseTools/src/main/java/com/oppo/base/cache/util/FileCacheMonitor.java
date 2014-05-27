/**
 * FileCacheMonitor.java
 * com.oppo.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-6-8 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache.util;

import java.io.File;
import java.util.concurrent.ConcurrentHashMap;

/**
 * ClassName:FileCacheMonitor
 * Function: 文件缓存监控, 删除过期的缓存文件
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-8  上午11:16:55
 */
public class FileCacheMonitor implements Runnable {

	private static FileCacheMonitor instance = new FileCacheMonitor();
	private final int INTERVEL = 5 * 60 * 1000; //5分钟进行一次检测
	private ConcurrentHashMap<String, Long> fileCacheMap;
	
	private FileCacheMonitor() {
		fileCacheMap = new ConcurrentHashMap<String, Long>(1024);
		Thread t = new Thread(this);
		t.setName("FileCacheMonitor");
		t.setDaemon(true);
		t.start();
	}
	
	public static FileCacheMonitor getInstance() {
		return instance;
	}
	
	/**
	 * 设置文件过期时间
	 * @param filePath
	 * @param cacheTime //缓存保存时间,以毫秒为单位
	 */
	public void setFileCacheTime(String filePath, long cacheTime) {
		if(cacheTime <= 0) { // 如果过期时间早于当前则删除
			removeFile(filePath);
		} else {
			fileCacheMap.put(filePath, System.currentTimeMillis() + cacheTime);
		}
	}
	
	private void removeFile(String filePath) {
		File f = new File(filePath);
		f.delete();
		
		fileCacheMap.remove(filePath);
	}
	
	public void run() {
		while(true) {
			long now = System.currentTimeMillis();
			for(String filePath : fileCacheMap.keySet()) {
				long expire = fileCacheMap.get(filePath);
				if(expire < now) {
					removeFile(filePath);
				}
			}
			
			try{
				Thread.sleep(INTERVEL);
			} catch(Exception e) {
			}
		}
	}
}

