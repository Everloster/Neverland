/**
 * MultiMemcacheMonitor.java
 * com.oppo.base.cache.util
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-26 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache.util;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.locks.ReentrantLock;

import com.oppo.base.cache.Memcache;

/**
 * ClassName:MultiMemcacheMonitor
 * Function: 具有关联key的memcache管理
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-26  上午09:16:38
 */
public class MultiMemcacheMonitor implements Runnable {
	//关联key管理实例
	private static MultiMemcacheMonitor instance = new MultiMemcacheMonitor();
	//监控线程
	private Thread monitorThread;
	//任务列表
	private Queue<MonitorTask> taskQueue;
	
	private static final ReentrantLock LOCK = new ReentrantLock();
	
	private static final int MAX_EMPTY_TRY = 1000;
	private static final int MIN_SLEEP_TIME = 50;
	private static final int MAX_SLEEP_TIME = 1000;
	
	private MultiMemcacheMonitor() {
		//初始化任务队列
		taskQueue = new LinkedList<MonitorTask>();
	}
	
	/**
	 * 获取缓存管理的实例
	 * @param 
	 * @return
	 */
	public static MultiMemcacheMonitor getInstance() {
		return instance;
	}
	
	/**
	 * 添加一个更新任务(需要等待后台执行实际的更新任务)
	 * @param multiKey 缓存管理key
	 * @param memcacheKey 缓存key
	 * @param expire 过期时间
	 * @param mem
	 * @return
	 */
	public boolean updateMultiKey(String multiKey, String memcacheKey, long expire, Memcache mem) {
		return this.updateMultiKey(multiKey, memcacheKey, expire, mem, false);
	}
	
	/**
	 * 添加一个更新任务(需要等待后台执行实际的更新任务)
	 * @param multiKey 缓存管理key
	 * @param memcacheKey 缓存key
	 * @param expire 过期时间
	 * @param mem
	 * @param waitForReturn 是否等执行完成后再返回
	 * @return
	 */
	public boolean updateMultiKey(String multiKey, String memcacheKey, long expire, Memcache mem, boolean waitForReturn) {
		MonitorTask mTask = new MonitorTask(multiKey, memcacheKey, expire, mem, MonitorTask.UPDATE);
		
		if(waitForReturn) {
			return this.handleTask(mTask);
		} else {
			this.addTask(mTask);
			return true;
		}
	}
	
	/**
	 * 添加一个删除任务(需要等待后台执行实际的删除任务)
	 * @param multiKey 缓存管理key
	 * @param memcacheKey 缓存key
	 * @param mem
	 * @return
	 */
	public boolean deleteMultiKey(String multiKey, String memcacheKey, Memcache mem) {
		return this.deleteMultiKey(multiKey, memcacheKey, mem, false);
	}
	
	/**
	 * 添加一个删除任务(需要等待后台执行实际的删除任务)
	 * @param multiKey 缓存管理key
	 * @param memcacheKey 缓存key
	 * @param mem
	 * @param waitForReturn 是否等执行完成后再返回
	 * @return
	 */
	public boolean deleteMultiKey(String multiKey, String memcacheKey, Memcache mem, boolean waitForReturn) {
		MonitorTask mTask = new MonitorTask(multiKey, memcacheKey, 0, mem, MonitorTask.UPDATE);
		
		if(waitForReturn) {
			return this.handleTask(mTask);
		} else {
			this.addTask(mTask);
			return true;
		}
	}
	
	@Override 
	public void run() {
		boolean isRunning = true;
		int emptyTry = 0;	//未获取到任务的次数
		int runTime = 0;	//总的循环次数
		while(isRunning) {
			MonitorTask mTask = null;
			//尝试获取任务
			LOCK.lock();
			try {
				int size = taskQueue.size();
				
				if(size > 0) {
					mTask = taskQueue.poll();
				}
			} finally {
				LOCK.unlock();
			}
			
			if(null != mTask) {
				boolean isSucc = this.handleTask(mTask);

				if(!isSucc) {
					addTask(mTask);
				}
				
				emptyTry = 0;
			} else {
				emptyTry++;
				//超过指定次数为获取到任务则停止线程
				if(emptyTry > MAX_EMPTY_TRY) {
					isRunning = false;
				}
			}
			
			runTime++;
			//50毫秒运行一次,每50次挂起1000毫秒
			int sleepTime = MIN_SLEEP_TIME;
			if(runTime % 50 == 0) {
				sleepTime = MAX_SLEEP_TIME;
				runTime = 0;
			}
			
			try {
				Thread.sleep(sleepTime);
			} catch(Exception ex){}
		}
	}
	
	/**
	 * 处理task
	 * @param 
	 * @return
	 */
	protected boolean handleTask(MonitorTask mTask) {
		if(mTask.getType() == MonitorTask.DELETE) {
			return doDeleteMultiKey(mTask);
		} else if(mTask.getType() == MonitorTask.UPDATE) {
			return doUpdateMultiKey(mTask);
		}
		
		return true;
	}
	
	/**
	 * 删除关联key的实际操作
	 * @param 
	 * @return
	 */
	protected boolean doDeleteMultiKey(MonitorTask mTask) {
		Memcache mem = mTask.getMemcache();
		
		String taskMultiKey = mTask.getMultiKey();
		Map<String, Long> multiKeys = null;

		try {
			multiKeys =	(Map<String, Long>)mem.get(taskMultiKey, null, 0);
			if(null != multiKeys) {
				int size = multiKeys.size();
				for(int i = 0; i < size; i++) {
					mem.delete(mTask.getMemcacheKey(), null, 0);
				}
			}
			
			//group key删除失败无影响，因此加上try - catch
			if(null != multiKeys) {
				try {
					mem.delete(taskMultiKey, null, 0);
				} catch(Exception ex){}
			} 
			
			return true;
		} catch (Exception e) {
			return false;
		}
	}
	
	/**
	 * 更新关联key的实际操作
	 * @param 
	 * @return
	 */
	protected boolean doUpdateMultiKey(MonitorTask mTask) {
		Memcache mem = mTask.getMemcache();
		
		String taskMultiKey = mTask.getMultiKey();
		Map<String, Long> multiKeys = null;
		
		try {
			multiKeys =	(Map<String, Long>)mem.get(taskMultiKey, null, 0);
			
			//新的缓存值
			String memcacheKey = mTask.getMemcacheKey();
			int size = (null == multiKeys) ? 0 : multiKeys.size();
			Map<String, Long> newMultiKeys = new HashMap<String, Long>(size + 1);
			
			//添加当前任务对应的key
			long now = System.currentTimeMillis();
			newMultiKeys.put(memcacheKey, mTask.getExpire() + now);
			
			if(null != multiKeys) {
				for(String key : multiKeys.keySet()) {
					if(memcacheKey.equals(key)) {
						continue;
					}
					
					long expire = multiKeys.get(key);
					if(expire > now) {
						newMultiKeys.put(key, expire);
					}
				}
			}
			
			//将multi key加入到memcache中
			return mem.setCache(taskMultiKey, newMultiKeys, 0);
		} catch (Exception e) {
			return false;
		}
	}

	//添加任务到队列
	private void addTask(MonitorTask mTask) {
		LOCK.lock();
		try {
			taskQueue.add(mTask);
			this.checkThread();
		} finally {
			LOCK.unlock();
		}
	}

	/**
	 * 检测线程是否启动，如果未启动则开启
	 * @param 
	 * @return
	 */
	private void checkThread() {
		if(null == monitorThread || !monitorThread.isAlive()) {
			//实例化为一个后台运行的线程并启动
			monitorThread = new Thread(this);
			monitorThread.setName("MultiMemcacheMonitor");
			monitorThread.setDaemon(true);
			monitorThread.setPriority(Thread.MIN_PRIORITY);
			monitorThread.start();
		}
	}
}