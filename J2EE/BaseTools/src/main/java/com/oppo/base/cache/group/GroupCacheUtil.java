/**
 * GroupCacheUtil.java
 * com.oppo.base.cache.group
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-25 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.cache.group;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.oppo.base.cache.DBConnect;
import com.oppo.base.cache.ICache;
import com.oppo.base.cache.Memcache;
import com.oppo.base.cache.StorageObject;

/**
 * ClassName:GroupCacheUtil
 * Function: 一组cache的查询
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-25  下午05:22:59
 */
public class GroupCacheUtil {
	private static final int MULTI_NUMBER = 15;
	
	/**
	 * 获取一组cache key对应值的方法，其中每个key对应一个Object
	 * @param cacheKeys cache key数组
	 * @param storageObjects cache key对应的数据读取实体
	 * @param timeout 数据过期时间
	 * @param cache 缓存处理类
	 * @param handler key联合处理类
	 * @return 数据list
	 * @throws Exception 
	 */
	public static <T> List<T> getListFromCache(String[] cacheKeys, StorageObject[] storageObjects, long timeout, ICache cache, IGroupHandler<T> handler) throws Exception {
		return getListFromCache(cacheKeys, storageObjects, timeout, cache, handler, null);
	}
	
	/**
	 * 获取一组cache key对应值的方法，其中每个key对应一个Object
	 * @param cacheKeys cache key数组
	 * @param storageObjects cache key对应的数据读取实体
	 * @param timeout 数据过期时间
	 * @param cache 缓存处理类
	 * @param handler key联合处理类
	 * @return 数据list
	 */
	public static <T> List<T> getListFromCache(String[] cacheKeys, StorageObject[] storageObjects, long timeout, ICache cache, IGroupHandler<T> handler, T defaultObj)
		throws Exception {
		Memcache mem = null;
		DBConnect db = null;

		//分离出memcache/db
		do {
			if(null == cache) {
				break;
			}
			
			if(cache instanceof Memcache) {
				mem = (Memcache)cache;
			} else if(cache instanceof DBConnect) {
				db = (DBConnect)cache;
			}
			
			cache = cache.getNextCache();
		} while(true);
		
		if(null == mem) {
			throw new Exception("cache must contains memcache");
		} else {
			mem.setNextCache(null);
		}
		
		int length = cacheKeys.length;
		//检查DB是否可用
		boolean isDBOpen = (null != db && !db.isTierClosed());
		//获取memcache中的数据
		List<T> cacheList = new ArrayList<T>(length);
		List<StorageObject> csList = getMultiCache(mem, cacheKeys, storageObjects, timeout, isDBOpen, cacheList);

		//没有需要组合的数据则直接返回
		if(null == csList || csList.size() == 0) {
			return cacheList;
		}

		//将无数据的进行合成
		List<T> dbList = (List<T>)db.get(null, handler.combin(csList), timeout);
		if(null != dbList) {
			if(!mem.isTierClosed()) {
				int dbListSize = dbList.size();
				for(int i = 0; i < dbListSize; i++) {
					T value = dbList.get(i);
					String key = handler.getRelateKey(value);
					mem.insert(key, new StorageObject("", value), timeout);
				}
			}
			
			cacheList.addAll(dbList);
		}
		
		return cacheList;
	}
	
	/**
	 * 获取一组cache key对应值的方法，其中每个key对应一个list
	 * @param cacheKeys cache key数组
	 * @param storageObjects cache key对应的数据读取实体
	 * @param timeout 数据过期时间
	 * @param cache 缓存处理类
	 * @param handler key联合处理类
	 * @return 一个map,其中key为cache key,value为cache key对应的list
	 */
	public static <T> Map<String, List<T>> getListMapFromCache(String[] cacheKeys, StorageObject[] storageObjects, long timeout, ICache cache, IGroupHandler<T> handler)
		throws Exception {
		Memcache mem = null;
		DBConnect db = null;

		//分离出memcache/db
		do {
			if(null == cache) {
				break;
			}
			
			if(cache instanceof Memcache) {
				mem = (Memcache)cache;
			} else if(cache instanceof DBConnect) {
				db = (DBConnect)cache;
			}
			
			cache = cache.getNextCache();
		} while(true);
		
		if(null == mem) {
			throw new Exception("cache must contains memcache");
		} else {
			mem.setNextCache(null);
		}
		
		int length = cacheKeys.length;
		//检查DB是否可用
		boolean isDBOpen = (null != db && !db.isTierClosed());
		//获取memcache中的数据
		Map<String, List<T>> cacheMap = new HashMap<String, List<T>>(length);
		List<StorageObject> csList = GroupCacheUtil.getMultiCache(mem, cacheKeys, storageObjects, timeout, isDBOpen, cacheMap);

		//没有需要组合的数据则直接返回
		if(null == csList || csList.size() == 0) {
			return cacheMap;
		}

		//将无数据的进行合成
		List<T> dbList = (List<T>)db.get(null, handler.combin(csList), timeout);
		if(null != dbList) {
			if(!mem.isTierClosed()) {
				//保存cache的map
				Map<String, List<T>> newCacheMap = new HashMap<String, List<T>>();
				
				int dbListSize = dbList.size();
				for(int i = 0; i < dbListSize; i++) {
					T value = dbList.get(i);
					String key = handler.getRelateKey(value);
					
					//拼出list
					List<T> tList = newCacheMap.get(key);
					if(null == tList) {
						tList = new ArrayList<T>();
					}
					tList.add(value);
					
					newCacheMap.put(key, tList);
				}
				
				//补充map中不存在的key
				for(int i = 0; i < length; i++) {
					String cacheKey = cacheKeys[i];
					if(cacheMap.containsKey(cacheKey)) {
						continue;
					}
					
					List<T> tList = newCacheMap.get(cacheKey);
					if(tList == null) {
						tList = new ArrayList<T>();
					}
					cacheMap.put(cacheKey, tList);
					mem.insert(cacheKey, new StorageObject("", tList), timeout);
				}
			}
		}
		
		return cacheMap;
	}
	
	/**
	 * 获取多个cacheSelector对应的缓存
	 * @param mem
	 * @param cacheKeys
	 * @param isDBOpen
	 * @param cacheList 缓存结果list，需要先初始化再传入，得到的结果将保存到此list中
	 * @return 没有获取到缓存的cacheSelector
	 */
	public static <T> List<StorageObject> getMultiCache(Memcache mem, String[] cacheKeys, 
			StorageObject[] storageObjects, long timeout, boolean isDBOpen, List<T> cacheList) throws Exception {
		int length = cacheKeys.length;
		
		List<StorageObject> csList = null;
		if(isDBOpen) {
			csList = new ArrayList<StorageObject>(length);
		}

		if(length <= MULTI_NUMBER) {
			//当缓存查询较少时,一个一个取
			for(int i = 0; i < length; i++) {
				T t = (T)mem.get(cacheKeys[i], null, timeout);
				if(null == t) {
					//获取未从memcache中获取到的数据
					if(isDBOpen) {
						csList.add(storageObjects[i]);
					}
				} else {
					cacheList.add(t);
				}
			}
		} else {
			//缓存查询较多时,组装cache key进行一次查询
			Object[] objs = mem.getCache(cacheKeys);
			for(int i = 0; i < length; i++) {
				T result = (T)objs[i];
				if(null == result) {
					if(isDBOpen) {
						csList.add(storageObjects[i]);
					}
				} else {
					cacheList.add(result);
				}
			}
		}
		
		return csList;
	}
	
	/**
	 * 获取多个cacheSelector对应的缓存
	 * @param mem
	 * @param cacheSelectors
	 * @param isDBOpen
	 * @param cacheMap 缓存结果map，需要先初始化再传入，得到的结果将保存到此map中
	 * @return 没有获取到缓存的cacheSelector
	 */
	public static <T> List<StorageObject> getMultiCache(Memcache mem, 
			String[] cacheKeys, StorageObject[] storageObjects, long timeout, 
			boolean isDBOpen, Map<String, List<T>> cacheMap) throws Exception {
		int length = cacheKeys.length;
		
		List<StorageObject> csList = null;
		if(isDBOpen) {
			csList = new ArrayList<StorageObject>(length);
		}
		
		if(length <= MULTI_NUMBER) {
			//当缓存查询较少时,一个一个取
			for(int i = 0; i < length; i++) {
				String cacheKey = cacheKeys[i];
				List<T> tList = (List<T>)mem.get(cacheKey, null, timeout);
				if(null == tList) {
					//获取未从memcache中获取到的数据
					if(isDBOpen) {
						csList.add(storageObjects[i]);
					}
				} else {
					cacheMap.put(cacheKey, tList);
				}
			}
		} else {
			//缓存查询较多时,组装cache key进行一次查询
			Object[] objs = mem.getCache(cacheKeys);
			for(int i = 0; i < length; i++) {
				List<T> resultList = (List<T>)objs[i];
				if(null == resultList) {
					if(isDBOpen) {
						csList.add(storageObjects[i]);
					}
				} else {
					cacheMap.put(cacheKeys[i], resultList);
				}
			}
		}
		
		return csList;
	}
}

