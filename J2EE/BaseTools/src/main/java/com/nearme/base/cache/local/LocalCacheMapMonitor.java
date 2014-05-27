/**
 * LocalCacheMapMonitor.java
 * com.nearme.base.cache.local
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-27 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache.local;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ClassName:LocalCacheMapMonitor <br>
 * Function: 本地的内存使用监控，当内存达到指定值时将会对本地缓存进行清理 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-27  下午03:55:08
 */
public class LocalCacheMapMonitor implements Runnable {
	private final ConcurrentHashMap<String, LocalCacheMap<String, CacheObject>> monitorMap;
	private long minFreeMemory;
	private final Logger logger = LoggerFactory.getLogger(LocalCacheMapMonitor.class);
	
	/**
	 * 每5秒钟进行一次扫描
	 */
	private static final long SLEEP_TIME = 5000; 
	
	private static LocalCacheMapMonitor instance;
	
	private LocalCacheMapMonitor(long minFreeMemory) {
		setMinFreeMemory(minFreeMemory);
		this.monitorMap = new ConcurrentHashMap<String, LocalCacheMap<String, CacheObject>>();
		
		initThread();
	}
	
	/**
	 * 获取LocalMapMonitor的实例
	 * @param 
	 * @return
	 */
	public synchronized static LocalCacheMapMonitor getInstance(long minFreeMemory) {
		if(null == instance) {
			instance = new LocalCacheMapMonitor(minFreeMemory);
		}
		
		return instance;
	}
	
	/**
	 * 是否包含指定名称的cache map
	 * @param 
	 * @return
	 */
	public boolean contains(String cacheMapName) {
		return this.monitorMap.containsKey(cacheMapName);
	}
	
	/**
	 * 将指定名称的cache map添加到监控中
	 * Function Description here
	 * @param 
	 * @return
	 */
	public void addCacheMap(String cacheMapName, LocalCacheMap<String, CacheObject> cacheMap) {
		//将需要处理的cache map放入监控map中
		this.monitorMap.putIfAbsent(cacheMapName, cacheMap);
	}
	
	public void run() {
		while(true) {
			//检测是否需要清理
			if(needClear()) {
				int totalRemoveCount = 0;
				//进行清理工作
				for(LocalCacheMap<String, CacheObject> cacheMap : monitorMap.values()) {
					totalRemoveCount += clearExpireCache(cacheMap);
				}
				
				if(totalRemoveCount > 0) {
					logger.info("本次共清理过期缓存个数:" + totalRemoveCount);
				} else {
					//需要清理时无过期元素
					logger.warn("无可清理的缓存,请检查缓存设置是否正确.");
				}
			}
			
			try {
				Thread.sleep(SLEEP_TIME);
			} catch (InterruptedException e) {
			}
		}
	}
	
	/**
	 * 是否需要清理cache
	 * @param 
	 * @return
	 */
	private boolean needClear() {
		//没有监控的则直接返回
		if(monitorMap.size() == 0) {
			return false;
		}
		
//		long max = memoryUsage.getMax();
//		long used = memoryUsage.getUsed();
		long free = Runtime.getRuntime().freeMemory();
		
		if(minFreeMemory > free) {
			logger.warn("最小可用内存设置:" + minFreeMemory + ",当前可用内存:" + free + ",准备清理缓存.");
			return true;
		} else {
			return false;
		}
	}
	
	/**
	 * 清理cache map中过期的值
	 * @param 
	 * @return 清理的过期值个数
	 */
	private int clearExpireCache(LocalCacheMap<String, CacheObject> cacheMap) {
		Map<String, CacheObject> map = cacheMap.cloneMap();
		int removeCount = 0;	//移除元素的个数
		for(String key : map.keySet()) {
			CacheObject co = map.get(key);
			if(co.isCacheExpire()) {
				cacheMap.remove(key);
				removeCount++;
			}
		}
		
		map.clear();
		map = null;
		
		return removeCount;
	}
	
	/**
	 * 监控线程初始化
	 * @param 
	 * @return
	 */
	private void initThread() {
		Thread monitorThread = new Thread(this);
		monitorThread.setName(this.getClass().getSimpleName() + monitorThread.getId());
		//设置为后台运行
		monitorThread.setDaemon(true);
		//设置为低优先级
		monitorThread.setPriority(4);
		//开始执行现场
		monitorThread.start();
	}
	
	private void setMinFreeMemory(long minFreeMemory) {
		this.minFreeMemory = minFreeMemory;
	}
}

