/**
 * CacheManager.java
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

import java.util.HashMap;

import com.danga.MemCached.SockIOPool;
import com.oppo.base.cache.DBConnect;
import com.oppo.base.cache.FileCache;
import com.oppo.base.cache.ICache;
import com.oppo.base.cache.Memcache;
import com.oppo.base.cache.config.CacheConfig;
import com.oppo.base.cache.config.CachePolicy;
import com.oppo.base.cache.exception.CacheConfigException;
import com.oppo.base.common.NumericUtil;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:CacheManager
 * Function: 缓存管理
 * 如果需要使用Memcache，则需要调用initCachePool初始化memcache参数
 * 如果需要使用File Cache，则需要调用setFileCacheDir设置缓存目录
 * 如果需要使用DB连接，则需要调用setSqlClient设置主从连接
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-8  上午11:26:33
 */
public class CacheManager {
	private static CacheManager instance = new CacheManager();
	private boolean isMemcachePoolInitial = false;
	
	private CacheManager() {
	}
	
	public static CacheManager getInstance() {
		return instance;
	}
	
	/**
	 * 初始化缓冲池,可设置的参数包括:
	 * PoolName,ServerList,ServerWeight,InitialConnection,MinConnection,
	 * MaxConnection,MaxIdelTime,MaxBusyTime,MaintSleep,SocketReadTimeout,
	 * SocketConnectTimeout,NagleAlgorithm,HashingAlgorithm,Failover
	 * 其中ServerList为必须设置的项，HashingAlgorithm在0-2之间
	 * 
	 * @param configMap
	 * @throws CacheConfigException
	 */
	public void initCachePool(HashMap<String, String> configMap) throws CacheConfigException{
		String poolName = configMap.get(Memcache.CONFIG_POOL_NAME);
		String serverListConfig = configMap.get(Memcache.CONFIG_SERVER_LIST);
		String serverWeightConfig = configMap.get(Memcache.CONFIG_SERVER_WEIGHT);
		String initConnConfig = configMap.get(Memcache.CONFIG_INIT_CONNECTION);
		String minConConfig = configMap.get(Memcache.CONFIG_MIN_CONNECTION);
		String maxConConfig = configMap.get(Memcache.CONFIG_MAX_CONNECTION);
		String maxIdelTimeConfig = configMap.get(Memcache.CONFIG_MAX_IDEL);
		String maxBusyTimeConfig = configMap.get(Memcache.CONFIG_MAX_BUSY);
		String maintSleepConfig = configMap.get(Memcache.CONFIG_MAINT_SLEEP);
		String socketTOConfig = configMap.get(Memcache.CONFIG_READ_TIMEOUT);
		String socketConnectTOConfig = configMap.get(Memcache.CONFIG_CONNECT_TIMEOUT);
		String nagleConfig = configMap.get(Memcache.CONFIG_NAGLE_ALGORITHM);
		String hashingAlgConfig = configMap.get(Memcache.CONFIG_HASHING_ALGORITHM);
		String failoverConfig = configMap.get(Memcache.CONFIG_FAILOVER);
        
		SockIOPool pool;
		if(StringUtil.isNullOrEmpty(poolName)) {
			pool = SockIOPool.getInstance();
		} else {
			pool = SockIOPool.getInstance(poolName);
		}
		
		if(StringUtil.isNullOrEmpty(serverListConfig)) {
			throw new CacheConfigException("server list can not be empty.");
		} else {
			pool.setServers(serverListConfig.split(","));
		}
		
		int hashingAlg = NumericUtil.parseInt(hashingAlgConfig, 2);
        if(hashingAlg > 2 || hashingAlg < 0) {
        	throw new CacheConfigException("Hashing Algorithm must between 0 and 2.");
        }
        pool.setHashingAlg(hashingAlg); 

		if(!StringUtil.isNullOrEmpty(serverWeightConfig)) {
			String[] serverWeight = serverWeightConfig.split(",");
			Integer[] weights = new Integer[serverWeight.length];
			for(int i = 0; i < weights.length; i++) {
				weights[i] = NumericUtil.parseInt(serverWeight[i], 1);
			}
			
			pool.setWeights(weights);
		}
		
		pool.setInitConn(NumericUtil.parseInt(initConnConfig, 10));
		pool.setMinConn(NumericUtil.parseInt(minConConfig, 5));
		pool.setMaxConn(NumericUtil.parseInt(maxConConfig, 50));
		pool.setMaxIdle(NumericUtil.parseInt(maxIdelTimeConfig, 1000 * 60 * 30));
        pool.setMaxBusyTime(NumericUtil.parseInt(maxBusyTimeConfig, 1000 * 60 * 5));   
        pool.setMaintSleep(NumericUtil.parseInt(maintSleepConfig, 1000 * 5));   
        pool.setSocketTO(NumericUtil.parseInt(socketTOConfig, 1000 * 3));   
        pool.setSocketConnectTO(NumericUtil.parseInt(socketConnectTOConfig, 1000 * 3));   
        pool.setNagle(Boolean.parseBoolean(nagleConfig)); 
        pool.setFailover(Boolean.parseBoolean(failoverConfig));
        
        pool.initialize();
        isMemcachePoolInitial = true;
	}
	
	/**
	 * 获取三层架构的缓存(memcache,file cache,db)
	 * @param type
	 * @param isWrite 是否为写模式
	 * @param config 缓存配置
	 * @return
	 */
	public ICache getThreeTierCache(String type, boolean isWrite, CacheConfig config) throws CacheConfigException {
		return getCache(type, 3, isWrite, DBConnect.SELECT_TYPE_ONE, config);
	}
	
	/**
	 * 获取三层架构的缓存(memcache,file cache,db)
	 * @param type
	 * @param isWrite 是否为写模式
	 * @param config 缓存配置
	 * @return
	 */
	public ICache getThreeTierCache(String type, boolean isWrite, int selectType, CacheConfig config) throws CacheConfigException {
		return getCache(type, 3, isWrite, selectType, config);
	}
	
	/**
	 * 获取DB连接
	 * @param isWrite 是否为写模式
	 * @param config
	 * @return
	 */
	public ICache getDBConnect(boolean isWrite, CacheConfig config) throws CacheConfigException {
		return getCache(null, 1, isWrite, DBConnect.SELECT_TYPE_ONE, config);
	}
	
	/**
	 * 获取DB连接
	 * @param isWrite 是否为写模式
	 * @param config
	 * @return
	 */
	public ICache getDBConnect(boolean isWrite, int selectType, CacheConfig config) throws CacheConfigException {
		return getCache(null, 1, isWrite, selectType, config);
	}
	
	/**
	 * @param type
	 * @param tier tier为1表示获取db层，为3表示获取三层
	 * @param isWrite 是否为写模式
	 * @return
	 * @throws CacheConfigException
	 */
	public ICache getCache(String type, int tier, boolean isWrite, int selectType, CacheConfig config) throws CacheConfigException {

		CachePolicy cp = config.getCachePolicy();
		
		DBConnect db = new DBConnect();
		db.setSelectType(selectType);
		db.setSqlSessionFactory(isWrite ? config.getSqlWriteFactory() : config.getSqlReadFactory());
		db.setTierClosed(cp.isDBClose());
		if(1 == tier) { // 如果只有1层
			if(isWrite && null == config.getSqlWriteFactory()) {
				throw new CacheConfigException("Sql write session can not be empty");
			} 
			
			if(!isWrite && null == config.getSqlReadFactory()) {
				throw new CacheConfigException("Sql read session can not be empty");
			}
			return db;
		}
		 
		FileCache fileCache = new FileCache();
		fileCache.setDirectory(config.getFileCacheDir() + type);
		fileCache.setNextCache(db);
		fileCache.setTierClosed(cp.isFileCacheClose());
		fileCache.setInsertSave(cp.isInsertSave());
		if(2 == tier) { 
			if(StringUtil.isNullOrEmpty(config.getFileCacheDir())) {
				throw new CacheConfigException("File cache directory can not be empty");
			}
			return fileCache;
		}
		
		Memcache mem = new Memcache();
		mem.setPoolName(config.getMemcachePoolName());
		mem.setNextCache(fileCache);
		mem.setTierClosed(cp.isMemcacheClose());
		mem.setInsertSave(cp.isInsertSave());
		mem.setPrefix(type);
		if(3 == tier) {
			if(!isMemcachePoolInitial){
				throw new CacheConfigException("SockIOPool is not initial");
			}
			return mem;
		}
		
		return null;
	}
	
	/**
	 * 获取不包含数据库层的缓存
	 * @param type
	 * @return
	 * @throws CacheConfigException
	 */
	public ICache getPureCache(String type, CacheConfig config) throws CacheConfigException {
		if(!isMemcachePoolInitial){
			throw new CacheConfigException("SockIOPool is not initial");
		}
		
		CachePolicy cp = config.getCachePolicy();
		
		FileCache fileCache = new FileCache();
		fileCache.setDirectory(config.getFileCacheDir() + type);
		fileCache.setTierClosed(cp.isFileCacheClose());
		fileCache.setInsertSave(cp.isInsertSave());
		
		Memcache mem = new Memcache();
		mem.setPoolName(config.getMemcachePoolName());
		mem.setNextCache(fileCache);
		mem.setTierClosed(cp.isMemcacheClose());
		mem.setInsertSave(cp.isInsertSave());
		mem.setPrefix(type);
		
		return mem;
	}
	
	/**
	 * 获取memcache
	 * @param 
	 * @return
	 */
	public ICache getMemcache(String type, CacheConfig config) throws CacheConfigException {
		if(!isMemcachePoolInitial){
			throw new CacheConfigException("SockIOPool is not initial.");
		}
		
		CachePolicy cp = config.getCachePolicy();
//		if(cp.isMemcacheClose()) {
//			throw new CacheConfigException("Memcache is not close.");
//		}
		
		Memcache mem = new Memcache();
		mem.setPoolName(config.getMemcachePoolName());
		mem.setPrefix(type);
		mem.setTierClosed(cp.isMemcacheClose());
		
		return mem;
	}
	
	/**
	 * 获取File Cache
	 * @param 
	 * @return
	 */
	public ICache getFileCache(String type, CacheConfig config) throws CacheConfigException {
		if(!isMemcachePoolInitial){
			throw new CacheConfigException("SockIOPool is not initial.");
		}
		
		CachePolicy cp = config.getCachePolicy();
//		if(cp.isFileCacheClose()) {
//			throw new CacheConfigException("File cache is not close.");
//		}
		
		FileCache fileCache = new FileCache();
		fileCache.setTierClosed(cp.isFileCacheClose());
		fileCache.setDirectory(config.getFileCacheDir() + type);
		
		return fileCache;
	}

	public static void main(String[] args) throws Exception {
		CacheManager cm = CacheManager.getInstance();
		HashMap<String, String> memcacheConfigMap = new HashMap<String, String>();
		memcacheConfigMap.put("ServerList", "127.0.0.1:11211");
		cm.initCachePool(memcacheConfigMap);
	}
}

