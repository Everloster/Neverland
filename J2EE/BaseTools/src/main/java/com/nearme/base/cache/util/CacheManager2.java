/**
 * CacheManager2.java
 * com.nearme.base.cache.util
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-19 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.ibatis.session.ExecutorType;

import com.danga.MemCached.SockIOPool;
import com.nearme.base.cache.DBConnect2;
import com.nearme.base.cache.MemFlagMemcache2;
import com.nearme.base.cache.Memcache2;
import com.nearme.base.cache.XmlFlagMemcache2;
import com.nearme.base.cache.selector.CacheSelector;
import com.oppo.base.cache.Memcache;
import com.oppo.base.cache.config.CacheConfig;
import com.oppo.base.cache.config.CachePolicy;
import com.oppo.base.cache.exception.CacheConfigException;
import com.oppo.base.common.NumericUtil;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:CacheManager2 <br>
 * Function: 缓存管理 <br>
 * 如果需要使用Memcache，则需要调用initCachePool初始化memcache参数 <br>
 * 如果需要使用DB连接，则需要调用setSqlClient设置主从连接 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-19  上午11:08:40
 */
public class CacheManager2 {
	private static CacheManager2 instance = new CacheManager2();
	private List<String> memcachePoolList;
	
	private CacheManager2() {
		memcachePoolList = new ArrayList<String>(16);
	}
	
	public static CacheManager2 getInstance() {
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
	public void initCachePool(Map<String, String> configMap) throws CacheConfigException{
		//从map中获取参数
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
        
		//设置参数
		SockIOPool pool;
		if(StringUtil.isNullOrEmpty(poolName)) {
			pool = SockIOPool.getInstance();
		} else {
			pool = SockIOPool.getInstance(poolName);
		}
		
		if(StringUtil.isNullOrEmpty(serverListConfig)) {
			throw new CacheConfigException("server list can not be empty.");
		} else {
			pool.setServers(StringUtil.split(serverListConfig, ','));
		}
		
		int hashingAlg = NumericUtil.parseInt(hashingAlgConfig, 2);
        if(hashingAlg > 2 || hashingAlg < 0) {
        	throw new CacheConfigException("Hashing Algorithm must between 0 and 2.");
        }
        pool.setHashingAlg(hashingAlg); 

		if(!StringUtil.isNullOrEmpty(serverWeightConfig)) {
			String[] serverWeight = StringUtil.split(serverWeightConfig, ',');
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
        
        //初始化
        pool.initialize();
        memcachePoolList.add(poolName);
	}
	
	/**
	 * 获取DB连接
	 * @param isWrite 是否为写模式
	 * @param selectType 
	 * @param config
	 * @return
	 */
	public DBConnect2 getDBConnect(boolean isWrite, int selectType, CacheConfig config) throws CacheConfigException {
		DBConnect2 db2 = new DBConnect2(selectType);
		db2.setSqlSessionFactory(isWrite ? config.getSqlWriteFactory() : config.getSqlReadFactory());
		
		db2.setTierClosed(config.getCachePolicy().isDBClose());
		
		return db2;
	}
	
	/**
	 * 获取DB连接
	 * @param isWrite 是否为写模式
	 * @param selectType
	 * @param executorType
	 * @param config
	 * @return
	 */
	public DBConnect2 getDBConnect(boolean isWrite, int selectType, ExecutorType executorType, CacheConfig config) throws CacheConfigException {
		DBConnect2 db2 = new DBConnect2(selectType, executorType);
		db2.setSqlSessionFactory(isWrite ? config.getSqlWriteFactory() : config.getSqlReadFactory());
		
		db2.setTierClosed(config.getCachePolicy().isDBClose());
		
		return db2;
	}
	
	/**
	 * 获取Memcache连接
	 * @param prefix memcache key前缀
	 * @param config
	 * @return
	 */
	public Memcache2 getMemcache(String prefix, CacheConfig config) throws CacheConfigException {
		return getMemcache(prefix, config, Memcache2.UPDATE_TYPE_NORMAL);
	}
	
	/**
	 * 获取Memcache连接
	 * @param prefix memcache key前缀
	 * @param config
	 * @param updateType 更新类型
	 * @return
	 */
	public Memcache2 getMemcache(String prefix, CacheConfig config, int updateType) throws CacheConfigException {
		String poolName = config.getMemcachePoolName();
		if(!this.isMemcachePoolInitial(poolName)) {
			throw new CacheConfigException(String.format("memcache[%s] empty", poolName));
		}
		
		CachePolicy cp = config.getCachePolicy();
		
		Memcache2 mem2 = null;
		//根据更新类型获取实例
		switch(updateType) {
		case Memcache2.UPDATE_TYPE_MEMCACHE:
			mem2 = new MemFlagMemcache2();
			break;
		case Memcache2.UPDATE_TYPE_XML:
			mem2 = new XmlFlagMemcache2();
			break;
		default:
			mem2 = new Memcache2();
		}

		mem2.setPoolName(poolName);
		mem2.setTierClosed(cp.isMemcacheClose());
		mem2.setPrefix(prefix);
		
		return mem2;
	}

	/**
	 * 指定poolName是否初始化过
	 * @param 
	 * @return
	 */
	protected boolean isMemcachePoolInitial(String poolName) {
		return memcachePoolList.contains(poolName);
	}

	public static void main(String[] args) throws Exception {
		CacheManager2 cm = CacheManager2.getInstance();
		//初始化memcache
		HashMap<String, String> memcacheConfigMap = new HashMap<String, String>();
		memcacheConfigMap.put("ServerList", "127.0.0.1:11211");
		cm.initCachePool(memcacheConfigMap);
		
		//获取缓存配置
		CacheConfig config = new CacheConfig();
		Memcache2 mem = cm.getMemcache("", config);
		mem.insert(new CacheSelector("1", "", "testvalue", 50000));
		System.out.println(mem.get(new CacheSelector("1")));
	}
}

