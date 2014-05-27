/**
 * MultiCPUTask.java
 * com.nearme.base.concurrent
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-5-9 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.concurrent;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * ClassName:MultiCPUTask <br>
 * Function: 利用多cpu进行任务处理 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-5-9  下午04:19:51
 */
public abstract class MultiCPUTask {
	/**
	 * 默认的线程数，为处理器个数
	 */
	public static final int DEFAULT_THREAD_COUNT = Runtime.getRuntime().availableProcessors();

	public static final ExecutorService DEFAULT_EXECUTOR = ExecutorManager.newCachedThreadPool("MultiCPUTask");

	private int maxThreadCount;
	private ExecutorService executor;

	public MultiCPUTask() {
		this(DEFAULT_EXECUTOR, DEFAULT_THREAD_COUNT);
	}

	public MultiCPUTask(ExecutorService executor) {
		this(executor, DEFAULT_THREAD_COUNT);
	}

	/**
	 * Creates a new instance of MultiCPUTask.
	 *
	 * @param executor 任务管理器，为null则在执行任务时自行创建及关闭
	 * @param threadCount 每次任务的最大线程数
	 *
	 */
	public MultiCPUTask(ExecutorService executor, int maxThreadCount) {
		this.executor = executor;
		this.maxThreadCount = maxThreadCount;
	}

	/**
	 * 任务处理
	 * @param dataList 被处理的数据列表
	 * @param minHandleSize 每个任务的最小处理数据条数
	 * @return
	 * @throws ExecutionException
	 * @throws InterruptedException
	 */
	public <T, D> List<D> handle(final List<T> dataList, int minHandleSize, final IMultiCPUHandler<T, D> handler) throws InterruptedException, ExecutionException {
		//这里不判断空了，由业务去判断, |-_-|
		int listSize = null == dataList ? 0 : dataList.size();
		if(listSize == 0) {
			return null;
		}

		//先将数据拆成n个模块
		int threadCount = DEFAULT_THREAD_COUNT;
		if(minHandleSize > 0) {
			threadCount = getDevideResult(listSize, minHandleSize);
		}

		//最大线程数不超过默认线程数
		if(threadCount > maxThreadCount) {
			threadCount = maxThreadCount;
		}

		if(threadCount == 1) {
			//如果只需一个线程则直接执行
			return doSingleHandle(dataList, handler);
		} else {
			//多线程执行
			return doMultiHandle(dataList, threadCount, handler);
		}
	}

	/**
	 *
	 * @param
	 * @return
	 */
	protected <T, D> List<D> doSingleHandle(final List<T> dataList, final IMultiCPUHandler<T, D> handler) {
		return handler.handle(dataList, 0, dataList.size() - 1);
	}

	/**
	 * 多线程任务执行
	 * @param dataList 被处理的数据列表
	 * @param threadCount 启用的线程数
	 * @param handler 执行计划任务的对象
	 * @return
	 */
	protected <T, D> List<D> doMultiHandle(final List<T> dataList, int threadCount, final IMultiCPUHandler<T, D> handler) throws InterruptedException, ExecutionException {
		//分配线程池，如果是此对象新建的则需要关闭
		boolean isNewExecutor = false;
		ExecutorService currentExecutor = this.getExecutor();
		if(null == currentExecutor) {
			currentExecutor = Executors.newFixedThreadPool(threadCount);
			isNewExecutor = true;
		}

		try {
			return doRealMultiHandle(dataList, currentExecutor, threadCount, handler);
		} finally {
			//关闭连接池
			if(isNewExecutor) {
				currentExecutor.shutdown();
			}
		}
	}

	/**
	 * 实际的任务处理及结果获取
	 * @param
	 * @return
	 */
	protected abstract <T, D> List<D> doRealMultiHandle(final List<T> dataList,
			ExecutorService currentExecutor,
			int threadCount,
			IMultiCPUHandler<T, D> handler) throws InterruptedException, ExecutionException;

	protected int getDevideResult(int dividend, int divisor) {
		return dividend / divisor + (dividend % divisor == 0 ? 0 : 1);
	}

	/**
	 * 获取maxThreadCount
	 * @return  the maxThreadCount
	 * @since   Ver 1.0
	 */
	public int getMaxThreadCount() {
		return maxThreadCount;
	}

	/**
	 * 设置maxThreadCount
	 * @param   maxThreadCount
	 * @since   Ver 1.0
	 */
	public void setMaxThreadCount(int maxThreadCount) {
		this.maxThreadCount = maxThreadCount;
	}

	/**
	 * 获取executor
	 * @return  the executor
	 * @since   Ver 1.0
	 */
	public ExecutorService getExecutor() {
		return executor;
	}

	/**
	 * 设置executor
	 * @param   executor
	 * @since   Ver 1.0
	 */
	public void setExecutor(ExecutorService executor) {
		this.executor = executor;
	}
}

