/**
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 * FileName:AbstractRunnableRecord.java
 * PackageName:com.oppo.base.record
 * Create Date:2011-4-6
 * History:
 *   ver	date	  author		desc
 * ────────────────────────────────────────────────────────
 *   1.0	2011-4-6	  80036381
 *
 *
*/

package com.oppo.base.record;

import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * ClassName:AbstractRunnableRecord
 * Function: 异步信息记录，相同业务只实例化一次，否则无效果
 * @author   80036381
 * @version  1.0
 * @since    Ver 1.0
 * @Date	 2011-4-6  下午08:25:32
 */
public abstract class AbstractRunnableRecord<T> implements IRecord<T>, Runnable {
	private long maxIdleSeconds;	//最大空闲时间
	private long millis;			//线程运行的休眠时间(毫秒)
	private int minHandleSize;		//最小处理大小，当队列大于等于此大小时，需要对信息进行持久化
	private Date lastWriteTime = new Date();	//最后写入时间
	private Queue<T> recordQueue;	//页面记录队列
	protected AtomicBoolean isThreadStart = new AtomicBoolean(false);//标记线程是否已经启动
	protected boolean isSystemShutdown = false;

	protected ReentrantLock lock = new ReentrantLock();//同步锁

	public AbstractRunnableRecord() {
		init();
	}

	/**
	 * 初始化参数
	 * @param
	 * @return
	 */
	public void init() {
		//默认空闲时间为5分钟
		maxIdleSeconds = 5 * 60 * 1000;
		//默认休眠时间为1秒
		millis = 1000;
		//默认大于等于1000条则处理
		minHandleSize = 1000;

		//创建队列
		recordQueue = newQueue();

		//初始化线程
		initThread();
	}

	protected Queue<T> newQueue() {
		return new LinkedList<T>();
	}

	/**
	 * 初始化线程
	 * @param
	 * @return
	 */
	protected boolean initThread() {
		if(!isThreadStart.compareAndSet(false, true)) {
			return false;
		}

		Thread recordThread = new Thread(this);
		recordThread.setName(this.getClass().getSimpleName() + recordThread.getId());
		//设置为后台运行
		recordThread.setDaemon(true);
		//设置为低优先级
		recordThread.setPriority(4);
		//开始执行现场
		recordThread.start();

		isSystemShutdown = false;

		return true;
	}

	/**
	 * 关闭任务
	 * @param
	 * @return
	 */
	public boolean shutdown() {
		if(isThreadStart.compareAndSet(true, false)) {
			isSystemShutdown = true;
		}

		return true;
	}

	public boolean isShutdown() {
		return isSystemShutdown;
	}

	public void run() {
		do {
			if(canRecord()) {
				//如果可以进行记录则开始记录
				int cycleTime = recordQueue.size() / this.getMinHandleSize();
				//至少执行一次
				if(cycleTime == 0) {
					cycleTime = 1;
				}
				for(int i = 0; i < cycleTime; i++) {
					if(record()){
						afterRecord();
					}
				}

				//更新最后处理时间
				lastWriteTime = new Date();
			}

			//每次执行后休眠一段时间
			try {
				Thread.sleep(this.getMillis());
			} catch (Exception ex) {}
		} while(!isSystemShutdown);
	}

	/**
	 * 添加一个记录任务
	 * @param tObj 需要记录的对象
	 */
	public boolean addRecordTask(T tObj) {
		//忽略空任务
		if(null == tObj) {
			return false;
		}

		//同步添加
		lock.lock();
		try {
			return recordQueue.add(tObj);
		} finally {
			lock.unlock();
		}
	}

	/**
	 * 添加一批记录任务
	 * @param tList 需要记录的对象集合
	 */
	public boolean addRecordTaskList(List<T> tList) {
		//忽略空任务
		if(null == tList || tList.size() == 0) {
			return false;
		}

		lock.lock();
		try {
			return recordQueue.addAll(tList);
		} finally {
			lock.unlock();
		}
	}

	/**
	 * 获取一个记录任务，同时将任务从队列中移除
	 * @param
	 * @return
	 */
	public T getRecordTask() {
		//同步获取
		lock.lock();
		try {
			return recordQueue.poll();
		} finally {
			lock.unlock();
		}
	}

	/**
	 * 默认超过一定时间则进行记录,或者记录数到达一定程度
	 */
	public boolean canRecord() {
		//获取当前时间与最后运行时间的间隔
		long interval = System.currentTimeMillis() - lastWriteTime.getTime();

		boolean canRun = interval > maxIdleSeconds;
		if(!canRun) {
			int size;

			lock.lock();
			try {
				size = recordQueue.size();
			} finally {
				lock.unlock();
			}

			canRun = size >= (isSystemShutdown ? 1 : getMinHandleSize());
		}

		return canRun;
	}

	/**
	 * 记录信息
	 *
	 */
	@Override
	public boolean record() {
		//每次处理指定大小的数量
		int handleSize = this.getMinHandleSize();

		clearCache();

		//备份list，如果写入失败则将导出的信息再添加回任务列表
		List<T> bakList = new ArrayList<T>(handleSize);
		while(handleSize-- > 0) {
			//获取记录对象
			T tObj = this.getRecordTask();

			if(null != tObj) {
				bakList.add(tObj);
				if(!addTaskToCache(tObj)) {
					//记录写入失败则将所有信息重新放回任务列表
					this.addRecordTaskList(bakList);
					bakList.clear();
					bakList = null;
					return false;
				}
			} else {
				//未获取到则结束获取
				break;
			}
		}

		if(bakList.size() > 0) {
			try {
				return saveDataFromCache();
			} catch(Exception e) {
				//记录写入失败则将所有信息重新放回任务列表
				this.addRecordTaskList(bakList);
				bakList.clear();
				bakList = null;

				error("save page access error.", e);
				return false;
			}
		} else {
			//不存在记录返回处理成功
			return true;
		}
	}

	/**
	 * 记录完成后需要进行的操作
	 * @param
	 * @return
	 */
	protected void afterRecord() {
	}

	protected Queue<T> getRecordQueue() {
		return recordQueue;
	}

	/**
	 * 将缓存中的值清除
	 * @param
	 * @return
	 */
	protected abstract void clearCache();

	/**
	 * 将要处理的值放到特定缓存中
	 * @param
	 * @return 是否添加成功，如果失败前面的操作应该回滚
	 */
	protected abstract boolean addTaskToCache(T tObj);

	/**
	 * 将特定缓存中的值持久化，如果失败前面的操作应该回滚
	 * @param
	 * @return 是否持久化成功
	 */
	protected abstract boolean saveDataFromCache() throws Exception;

	/**
	 * 记录错误信息
	 * @param
	 * @return
	 */
	protected abstract void error(String info, Throwable throwable);

	/**
	 * 获取最大空闲时间
	 * @return  the maxIdleSeconds
	 * @since   CodingExample Ver 1.0
	 */
	public long getMaxIdleSeconds() {
		return maxIdleSeconds;
	}

	/**
	 * 设置最大空闲时间
	 * @param   maxIdleSeconds    the maxIdleSeconds to set
	 * @since   CodingExample Ver 1.0
	 */

	public void setMaxIdleSeconds(long maxIdleSeconds) {
		this.maxIdleSeconds = maxIdleSeconds;
	}

	/**
	 * 获取休眠时间,以毫秒为单位
	 * @return  the millis
	 * @since   CodingExample Ver 1.0
	 */

	public long getMillis() {
		return millis;
	}

	/**
	 * 设置休眠时间,以毫秒为单位
	 * @param   millis    the millis to set
	 * @since   CodingExample Ver 1.0
	 */

	public void setMillis(long millis) {
		this.millis = millis;
	}

	/**
	 * 设置最小处理大小，当队列大于等于此大小时，需要对信息进行持久化
	 * @param   minHandleSize
	 * @since   Ver 1.0
	 */
	public void setMinHandleSize(int minHandleSize) {
		this.minHandleSize = minHandleSize;
	}

	/**
	 * 最小处理大小，当队列大于等于此大小时，需要对信息进行持久化
	 * @param
	 * @return
	 */
	protected int getMinHandleSize() {
		if(this.minHandleSize <= 0) {
			this.minHandleSize = 1000;
		}

		return this.minHandleSize;
	}
}

