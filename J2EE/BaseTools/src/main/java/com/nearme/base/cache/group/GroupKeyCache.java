/**
 * GroupKeyCache.java
 * com.nearme.base.cache.group
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-25 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache.group;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.nearme.base.cache.DBConnect2;
import com.nearme.base.cache.ICache2;
import com.nearme.base.cache.Memcache2;
import com.nearme.base.cache.selector.CacheSelector;

/**
 * ClassName:GroupKeyCache
 * Function: 一组cache的查询
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-25  下午03:37:20
 */
public class GroupKeyCache {
	/**
	 * 当一次缓存获取超过了这个数时，采取getMulti的方式获取缓存
	 */
	private static final int MULTI_NUMBER = 15;
	
	/**
	 * 获取一组cache selector对应值的方法，其中每个key对应一个Object
	 * @param cacheSelectors 需要查询的cache selector集合
	 * @param cache 用来查询的缓存，必须保证cache中包含memcache
	 * @param handler 针对一组cache selector进行处理的接口
	 * @return
	 */
	public static <T> List<T> getListFromCache(CacheSelector[] cacheSelectors, ICache2 cache, IGroupKeyHandler<T> handler)
		throws Exception {
		Memcache2 mem = null;
		DBConnect2 db = null;
		
		//分离出memcache/db
		do {
			if(null == cache) {
				break;
			}
			
			if(cache instanceof Memcache2) {
				mem = (Memcache2)cache;
			} else if(cache instanceof DBConnect2) {
				db = (DBConnect2)cache;
			}
			
			cache = cache.getNextCache();
		} while(true);
		
		if(null == mem) {
			throw new Exception("cache must contains memcache");
		} else {
			mem.setNextCache(null);
		}
		
		//
		int length = cacheSelectors.length;
		boolean isDBOpen = (null != db && !db.isTierClosed());

		//获取memcache中的数据
		List<T> cacheList = new ArrayList<T>(length);
		List<CacheSelector> csList = getMultiCache(mem, cacheSelectors, isDBOpen, cacheList);
		
		//直接返回
		if(null == csList || csList.size() == 0) {
			return cacheList;
		}
		
		//一次查询得到所有数据
		db.setSelectType(DBConnect2.SELECT_TYPE_LIST);
		List<T> dbList = db.get(handler.combin(csList));

		if(null != dbList) {
			if(!mem.isTierClosed()) {
				int dbListSize = dbList.size();
				
				//将db中获取到的值设置到memcache中
				Map<String, Integer> containMap = new HashMap<String, Integer>(dbListSize);//含值的cache
				for(int i = 0; i < dbListSize; i++) {
					T value = dbList.get(i);
					CacheSelector cs = handler.getRelateKey(value);
					cs.setSelectorValue(value);
					mem.insert(cs);
					
					containMap.put(cs.getCacheKey(), 1);
				}
				
				//补充未找到值的cache(保护emptyValue的)
				Map<String, CacheSelector> waitingMap = transToMap(csList);	//待处理的cache
				for(String key : waitingMap.keySet()) {
					if(!containMap.containsKey(key)) {
						//未找到则补充
						mem.insert(waitingMap.get(key));
					}
				}
			}
				
			cacheList.addAll(dbList);
		}
		
		csList.clear();
		csList = null;

		return cacheList;
	}
	
	/**
	 * 获取一组cache selector对应值的方法，其中每个key对应一个list
	 * @param cacheSelectors 需要查询的cache selector集合
	 * @param cache 用来查询的缓存，必须保证cache中包含memcache
	 * @param handler 针对一组cache selector进行处理的接口
	 * @return
	 */
	public static <T> Map<String, List<T>> getListMapFromCache(CacheSelector[] cacheSelectors, ICache2 cache, IGroupKeyHandler<T> handler)
		throws Exception {
		Memcache2 mem = null;
		DBConnect2 db = null;
		
		//分离出memcache/db
		do {
			if(null == cache) {
				break;
			}
			
			if(cache instanceof Memcache2) {
				mem = (Memcache2)cache;
			} else if(cache instanceof DBConnect2) {
				db = (DBConnect2)cache;
			}
			
			cache = cache.getNextCache();
		} while(true);
		
		if(null == mem) {
			throw new Exception("cache must contains memcache");
		} else {
			mem.setNextCache(null);
		}
		
		//
		int length = cacheSelectors.length;
		boolean isDBOpen = (null != db && !db.isTierClosed());
		
		//获取memcache中的数据
		Map<String, List<T>> cacheMap = new HashMap<String, List<T>>(length);
		List<CacheSelector> csList = getMultiCache(mem, cacheSelectors, isDBOpen, cacheMap);
		
		//直接返回
		if(null == csList || csList.size() == 0) {
			return cacheMap;
		}
		
		//一次查询得到所有数据
		db.setSelectType(DBConnect2.SELECT_TYPE_LIST);
		List<T> dbList = db.get(handler.combin(csList));
		csList.clear();
		csList = null;

		//将获取到的数据设置到memcache中
		if(null != dbList) {
			if(!mem.isTierClosed()) {
				int dbListSize = dbList.size();
				//保存cache的map
				Map<String, List<T>> newCacheMap = new HashMap<String, List<T>>(dbListSize);
				
				long timeout = 0;
				for(int i = 0; i < dbListSize; i++) {
					T value = dbList.get(i);
					CacheSelector cs = handler.getRelateKey(value);
					cs.setSelectorValue(value);
					timeout = cs.getTimeout();
					
					//拼出list
					String cacheKey = cs.getCacheKey();
					List<T> tList = newCacheMap.get(cacheKey);
					if(null == tList) {
						tList = new ArrayList<T>();
					}
					tList.add(value);
					
					newCacheMap.put(cacheKey, tList);
				}
				
				//补充cache map中不存在的key，并将其设置到缓存中
				for(int i = 0; i < length; i++) {
					String cacheKey = cacheSelectors[i].getCacheKey();
					//如果已经缓存中包含此值则跳过
					if(cacheMap.containsKey(cacheKey)) {
						continue;
					}
					
					List<T> tList = newCacheMap.get(cacheKey);
					//如果为null则默认为空的list
					if(null == tList) {
						tList = new ArrayList<T>(0);
					}
					
					mem.insert(new CacheSelector(cacheKey, null, tList, timeout));
					cacheMap.put(cacheKey, tList);
				}
				
				newCacheMap.clear();
				newCacheMap = null;
			}
		}

		return cacheMap;
	}
	
	/**
	 * 获取多个cacheSelector对应的缓存
	 * @param mem
	 * @param cacheSelectors
	 * @param isDBOpen
	 * @param cacheList 缓存结果list，需要先初始化再传入，得到的结果将保存到此list中
	 * @return 没有获取到缓存的cacheSelector
	 */
	public static <T> List<CacheSelector> getMultiCache(Memcache2 mem, CacheSelector[] cacheSelectors, 
			boolean isDBOpen, List<T> cacheList) throws Exception {
		int length = cacheSelectors.length;
		
		List<CacheSelector> csList = null;
		if(isDBOpen) {
			csList = new ArrayList<CacheSelector>(length);
		}

		if(length <= MULTI_NUMBER) {
			//当缓存查询较少时,一个一个取
			for(int i = 0; i < length; i++) {
				CacheSelector cacheSelector = cacheSelectors[i];
				T t = (T)mem.getCache(cacheSelector);
				if(null == t) {
					//获取未从memcache中获取到的数据
					if(isDBOpen) {
						csList.add(cacheSelector);
					}
				} else {
					cacheList.add(t);
				}
			}
		} else {
			//缓存查询较多时,组装cache key进行一次查询
			String[] cacheKeys = new String[length];
			for(int i = 0; i < length; i++) {
				cacheKeys[i] = cacheSelectors[i].getCacheKey();
			}
			
			Object[] objs = mem.getCache(cacheKeys);
			for(int i = 0; i < length; i++) {
				T result = (T)objs[i];
				if(null == result) {
					if(isDBOpen) {
						csList.add(cacheSelectors[i]);
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
	public static <T> List<CacheSelector> getMultiCache(Memcache2 mem, CacheSelector[] cacheSelectors, 
			boolean isDBOpen, Map<String, List<T>> cacheMap) throws Exception {
		int length = cacheSelectors.length;
		
		List<CacheSelector> csList = null;
		if(isDBOpen) {
			csList = new ArrayList<CacheSelector>(length);
		}
		
		if(length <= MULTI_NUMBER) {
			//当缓存查询较少时,一个一个取
			for(int i = 0; i < length; i++) {
				CacheSelector cacheSelector = cacheSelectors[i];
				List<T> tList = (List<T>)mem.getCache(cacheSelector);
				if(null == tList) {
					//获取未从memcache中获取到的数据
					if(isDBOpen) {
						csList.add(cacheSelector);
					}
				} else {
					cacheMap.put(cacheSelector.getCacheKey(), tList);
				}
			}
		} else {
			//缓存查询较多时,组装cache key进行一次查询
			String[] cacheKeys = new String[length];
			for(int i = 0; i < length; i++) {
				cacheKeys[i] = cacheSelectors[i].getCacheKey();
			}
			
			Object[] objs = mem.getCache(cacheKeys);
			for(int i = 0; i < length; i++) {
				List<T> resultList = (List<T>)objs[i];
				if(null == resultList) {
					if(isDBOpen) {
						csList.add(cacheSelectors[i]);
					}
				} else {
					cacheMap.put(cacheSelectors[i].getCacheKey(), resultList);
				}
			}
		}
		
		return csList;
	}
	
	/**
	 * 将存在空值映射的cache列表转换为map，map的key为缓存的key
	 * @param 
	 * @return 
	 */
	private static Map<String, CacheSelector> transToMap(List<CacheSelector> csList) {
		int len = csList.size();
		Map<String, CacheSelector> map = new HashMap<String, CacheSelector>(len);
		for(int i = 0; i < len; i++) {
			CacheSelector cs = csList.get(i);
			
			if(null != cs.getEmptyValue()) {
				cs.setSelectorValue(cs.getEmptyValue());
				map.put(cs.getCacheKey(), cs);
			}
		}
		
		return map;
	}
}

