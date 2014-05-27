/**
 * ExecutorManager.java
 * com.nearme.base.concurrent
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-4-2 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.gamecenter.ddz.oauth.tool.concurrent;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

/**
 * ClassName:ExecutorManager <br>
 * Function: 线程池的管理 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-4-2  下午1:12:07
 */
public final class ExecutorManager {

	/**
	 * 存放所有的线程池进行管理
	 */
	private static final ConcurrentLinkedQueue<ExecutorService> EXECUTORS_QUEUE =
			new ConcurrentLinkedQueue<ExecutorService>();

	/**
	 * 创建一个可根据需要创建新线程的线程池
	 * @param
	 * @return
	 */
	public static ExecutorService newCachedThreadPool() {
		ExecutorService executorService = Executors.newCachedThreadPool();
		manageExecutorService(executorService);

		return executorService;
	}


	/**
	 * 创建一个可根据需要创建新线程的线程池,在创建线程池时根据指定名称命名
	 * Function Description here
	 * @param
	 * @return
	 */
	public static ExecutorService newCachedThreadPool(String name) {
		return newCachedThreadPool(new NamedThreadFactory(name));
	}

	/**
	 * 创建一个可根据需要创建新线程的线程池,在创建线程池时根据ThreadFactory创建
	 * Function Description here
	 * @param
	 * @return
	 */
	public static ExecutorService newCachedThreadPool(ThreadFactory threadFactory) {
		ExecutorService executorService = Executors.newCachedThreadPool(threadFactory);
		manageExecutorService(executorService);

		return executorService;
	}

	/**
	 * 创建一个指定大小的线程池
	 * @param
	 * @return
	 */
	public static ExecutorService newFixedThreadPool(int nThreads) {
		ExecutorService executorService = Executors.newFixedThreadPool(nThreads);
		manageExecutorService(executorService);

		return executorService;
	}

	/**
	 * 创建一个指定大小的线程池，在创建线程时根据ThreadFactory创建
	 * @param
	 * @return
	 */
	public static ExecutorService newFixedThreadPool(int nThreads, ThreadFactory threadFactory) {
		ExecutorService executorService = Executors.newFixedThreadPool(nThreads, threadFactory);
		manageExecutorService(executorService);

		return executorService;
	}



	/**
	 * 加入指定的ExecutorService进行统一管理
	 * @param
	 * @return
	 */
	public static void manageExecutorService(ExecutorService executorService) {
		EXECUTORS_QUEUE.offer(executorService);
	}

	/**
	 * 关闭指定线程池
	 * @param
	 * @return
	 */
	public static boolean shutdown(ExecutorService executorService) {
		EXECUTORS_QUEUE.remove(executorService);
		return doShutdown(executorService);
	}

	/**
	 * 关闭所有被管理起来的线程池
	 * @param
	 * @return
	 */
	public static void shutdownAll() {
		ExecutorService es = null;
		while(null != (es = EXECUTORS_QUEUE.poll())) {
			doShutdown(es);
		}
	}

	/**
	 * 关闭线程池的实际操作
	 * @param
	 * @return
	 */
	private static boolean doShutdown(ExecutorService executorService) {
		if(executorService.isShutdown()) {
			return true;
		} else {
			try {
				executorService.shutdownNow();
	        } catch (SecurityException ex) {
	        	try {
	        		executorService.shutdown();
	        	} catch (SecurityException ex1) {
	        		return false;
	        	}
	        }


			return true;
		}
	}
}

