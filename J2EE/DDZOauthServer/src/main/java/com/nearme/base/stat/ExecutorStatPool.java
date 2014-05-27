/**
 * ExecutorStatPool.java
 * com.nearme.base.stat
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-12-12 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.stat;

import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.oppo.base.time.TimeFormat;

/**
 * ClassName:ExecutorStatPool <br>
 * Function: 执行统计实体生成类,当日志级别小于等于info时可打印出结果 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-12-12  上午09:54:43
 */
public class ExecutorStatPool implements Runnable {
	
	public static final String LOGGER_NAME = "com.nearme.stat.ExecutorStat";
	public static final String STAT_POOL_START = "=====================stat output at ";
	public static final String STAT_POOL_END = "=====================";
	
	private static final ExecutorStatPool INSTANCE = new ExecutorStatPool();
	
	private Logger logger;	//日志记录对象
	private long interval;	//记录线程执行间隔
	private boolean resetOld; //每次输出后是否重置数据
	
	/**
	 * 存放已有实体的map
	 */
	private final ConcurrentHashMap<String, ExecutorStat> executorStatMap = new ConcurrentHashMap<String, ExecutorStat>();
	
	private ExecutorStatPool() {
		init();
		initThread();
	}
	
	/**
	 * 获取统计池
	 * @param 
	 * @return
	 */
	public static ExecutorStatPool getInstance() {
		return INSTANCE;
	}
	
	/**
	 * 根据标识符获取执行统计实体
	 * @param 
	 * @return
	 */
	public ExecutorStat getExecutorStat(String identifier) {
		return getExecutorStat(identifier, 0);
	}
	
	/**
	 * 根据标识符获取执行统计实体
	 * @param 
	 * @return
	 */
	public ExecutorStat getExecutorStat(String identifier, int childSize) {
		String key = identifier + "_" + childSize;
		ExecutorStat execStat = executorStatMap.get(key);
		if(null == execStat) {
			execStat = new ExecutorStat(identifier, childSize);
			ExecutorStat oldExecStat = executorStatMap.putIfAbsent(key, execStat);
			if(null != oldExecStat) {
				execStat = oldExecStat;
			}
		}
			
		return execStat;
	}
	
	public void run() {
		while(true) {
			//每隔一定时间生成统计日志
			try {
				Thread.sleep(interval);
			} catch (InterruptedException e) {
			}

			output();
		}
	}
	
	/**
	 * 将统计信息输出，如果要改变其输出方式，可以重写此方法
	 * @param execStat
	 * @return
	 */
	protected void output() {
		if (null == logger || !logger.isInfoEnabled()
				|| executorStatMap.isEmpty()) {
			return;
		}

		logger.info(STAT_POOL_START + TimeFormat.getCurrentTime() + STAT_POOL_END);
		for(ExecutorStat execStat : executorStatMap.values()) {
			String statInfo = execStat.toStatString(resetOld);
			if(null != statInfo) {
				logger.info(statInfo);
			}
		}
	}
	
	/**
	 * 参数初始化，包括日志记录类及统计执行时间间隔
	 * @param 
	 * @return
	 */
	protected void init() {
//		logger = LoggerFactory.getLogger(LOGGER_NAME);
		interval = 5 * 60 * 1000;	//默认5分钟执行一次
		resetOld = true;			//默认每次统计完后重置
	}
	
	//开启数据记录的线程
	private void initThread() {
		Thread logThread = new Thread(this);
		logThread.setName(this.getClass().getSimpleName() + logThread.getId());
		//设置为后台运行
		logThread.setDaemon(true);
		//设置为低优先级
		logThread.setPriority(4);
		//开始执行现场
		logThread.start();
	}

	/**
	 * 获取日志处理类
	 * @return  the logger
	 * @since   Ver 1.0
	 */
	public Logger getLogger() {
		return logger;
	}

	/**
	 * 设置日志处理类
	 * @param   logger    
	 * @since   Ver 1.0
	 */
	public void setLogger(Logger logger) {
		this.logger = logger;
	}

	/**
	 * 获取记录统计信息的频率
	 * @return  the interval
	 * @since   Ver 1.0
	 */
	public long getInterval() {
		return interval;
	}

	/**
	 * 设置记录统计信息的频率
	 * @param   interval    
	 * @since   Ver 1.0
	 */
	public void setInterval(long interval) {
		if(interval <= 0) {
			return;
		}
		
		this.interval = interval;
	}

	/**
	 * 获取每次输出后是否重置数据
	 * @return  the resetOld
	 * @since   Ver 1.0
	 */
	public boolean isResetOld() {
		return resetOld;
	}

	/**
	 * 设置每次输出后是否重置数据
	 * @param   resetOld    
	 * @since   Ver 1.0
	 */
	public void setResetOld(boolean resetOld) {
		this.resetOld = resetOld;
	}
	
	public static void main(String[] args) {
		ExecutorStatPool pool = ExecutorStatPool.getInstance();
		pool.setLogger(LoggerFactory.getLogger("test"));

		ExecutorStat es0 = pool.getExecutorStat("test0");
		es0.addStat(true, 1000);
		es0.addStat(true, 1023);
		es0.addStat(true, 1010);
		es0.addStat(true, 1020);
		es0.addStat(false, 1111);
		
		ExecutorStat es1 = pool.getExecutorStat("test1");
		es1.addStat(true, 1000);
		es1.addStat(true, 1023);
		es1.addStat(true, 1010);
		es1.addStat(true, 1020);
		es1.addStat(false, 1111);
		
		pool.output();
	}
}

