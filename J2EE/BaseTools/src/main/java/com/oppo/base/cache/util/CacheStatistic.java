package com.oppo.base.cache.util;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

import com.oppo.base.common.NumericUtil;
import com.oppo.base.common.StringUtil;
import com.oppo.base.file.FileOperate;

/**
 * ClassName:CacheStatistic
 * Function: 统计各层缓存命中次数,记录在内存中,程序重启则丢失
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-8  上午10:58:29
 */
public class CacheStatistic {
	private static CacheStatistic instance = new CacheStatistic();
	private static final String TYPE_SPLIT = "\n";
	private static final String VALUE_SPLIT = ":";
	
	private Map<String, Long> recordMap; 	//统计map
	private int totalCount;		//总数
	private volatile boolean isOutput;	//是否已经输出
	private ReentrantLock lock = new ReentrantLock();//同步锁
	private volatile long lastRuntime;	//最后一次运行时间
	
	//单例
	private CacheStatistic(){
		lastRuntime = System.currentTimeMillis();
		totalCount = 0;
		recordMap = new ConcurrentHashMap<String, Long>();
	}
	
	public static CacheStatistic getInstance() {
		return instance;
	}
	
	/**
	 * 添加对应缓存的命中次数，每次加1
	 * @param key 缓存名
	 * @return
	 */
	public void addHit(String key) {
		addHit(key , 1);
	}
	
	/**
	 * 添加对应缓存的命中次数
	 * @param key 缓存名
	 * @param hitCount 命中次数
	 * @return
	 */
	public void addHit(String key, long hitCount) {
		//加锁
		lock.lock();
		
		try {
			totalCount += hitCount;
			if(recordMap.containsKey(key)) {
				hitCount += recordMap.get(key);
			}
			recordMap.put(key, hitCount);
		} finally {
			//释放锁
			lock.unlock();
		}
	}
	
	/**
	 * 获取添加对应缓存的命中次数
	 * @param key 缓存名
	 * @return
	 */
	public long getHit(String key) {
		if(recordMap.containsKey(key)) {
			return recordMap.get(key);
		} else {
			return 0;
		}
	}
	
	public void clear() {
		recordMap.clear();
	}
	
	/**
	 * 输出结果到指定文件中
	 * @param path
	 */
	public void output(String path) throws Exception {
		long now = System.currentTimeMillis();
		//无数据或者上次统计时间间隔小于5分钟则暂不输出
		if(0 == recordMap.size() || now - lastRuntime < 5 * 60 * 1000) {
			return;
		}
		
		lastRuntime = now;
		
		//第一次输出前考虑和之前的合并
		if(!isOutput) {
			String content = FileOperate.getFileContent(path, null);
			if(!StringUtil.isNullOrEmpty(content)) {
				String[] keySec = content.split(VALUE_SPLIT);
				if(keySec.length == 2) {
					this.addHit(keySec[0], NumericUtil.parseLong(keySec[1], 0));
				}
			}
			
			isOutput = true;
		}
		
		StringBuilder sb = new StringBuilder();
		for(String key : recordMap.keySet()) {
			sb.append(key).append(VALUE_SPLIT).append(recordMap.get(key)).append(TYPE_SPLIT);
		}
		
		FileOperate.saveContentToFile(path, sb.toString(), null);
	}
}
