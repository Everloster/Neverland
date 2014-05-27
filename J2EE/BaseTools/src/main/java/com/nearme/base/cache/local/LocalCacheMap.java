/**
 * LocalCacheMap.java
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

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * ClassName:LocalCacheMap <br>
 * Function: 本地缓存 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-27  下午03:35:07
 */
public class LocalCacheMap<K, V> extends LinkedHashMap<K, V> {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	private final int maxCapacity;	//map的最大容量，超过这个量则将开始清除，小于1则不按此处理
	private long minFreeMemory;
		
	private static final float DEFAULT_LOAD_FACTOR = 0.75f;
	
	private ReentrantReadWriteLock lock;
	private ReentrantReadWriteLock.WriteLock writeLock;	//写锁
	private ReentrantReadWriteLock.ReadLock readLock;	//读锁
	
	/**
	 * @param maxCapacity map的最大容量，如果小于等于0则不限制
	 */
	public LocalCacheMap(int maxCapacity) {
		this(maxCapacity, 0, true);
	}
	
	/**
	 * @param maxCapacity map的最大容量，如果小于等于0则不限制
	 * @param minFreeMemory 最小空闲内存，小于该内存会进行内存清理
	 */
	public LocalCacheMap(int maxCapacity, long minFreeMemory) {
		this(maxCapacity, minFreeMemory, true);
	}
	
	/**
	 * @param maxCapacity map的最大容量，如果小于等于0则不限制
	 * @param accessOrder 访问方式
	 */
	public LocalCacheMap(int maxCapacity, long minFreeMemory, boolean accessOrder) {
		super(maxCapacity, DEFAULT_LOAD_FACTOR, accessOrder);
		
		this.maxCapacity = maxCapacity;
		this.minFreeMemory = minFreeMemory;
		 
		//创建读写锁
		lock = new ReentrantReadWriteLock();
		writeLock = lock.writeLock();
		readLock = lock.readLock();
	}
	 
	@Override    
    public V get(Object key) {    
        try {    
        	readLock.lock();
        	return super.get(key);    
        } finally {    
        	readLock.unlock();    
        }    
    }    
     
    @Override    
    public V put(K key, V value) {
        try {
        	writeLock.lock();
        	return super.put(key, value);
        } finally {
        	writeLock.unlock();
        }
    }
    
    public Map<K, V> cloneMap() {
        try {
        	readLock.lock();
        	return new HashMap<K, V>(this);
        } finally {
        	readLock.unlock();
        }
    }
    
    @Override
    protected boolean removeEldestEntry(java.util.Map.Entry<K, V> eldest) {
    	if(maxCapacity > 0 && size() > maxCapacity) {
    		//map元素个数大于最大数则清理
    		return true;
    	} else if(minFreeMemory > 0 && Runtime.getRuntime().freeMemory() < minFreeMemory) {
    		//空闲内存小于最小内存则清理
    		return true;
    	} else {
    		return false;
    	}
    }
}

