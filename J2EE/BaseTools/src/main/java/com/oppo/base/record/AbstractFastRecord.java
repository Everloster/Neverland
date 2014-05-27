/**
 * AbstractFastRecord.java
 * com.oppo.base.record
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-20 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.oppo.base.record;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * ClassName:AbstractFastRecord <br>
 * Function: 快速记录，存在数据时就通知数据写入 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-20  上午9:48:36
 */
public abstract class AbstractFastRecord<T> extends AbstractRunnableRecord<T> {

	private ConcurrentLinkedQueue<T> queue;

	public void run() {
		do {
			while (queue.isEmpty()) {
				try {
					synchronized (queue) {
//						System.out.println("wait...");
						queue.wait();
					}
				} catch (InterruptedException e) {
				}
			}

//			System.out.println("run...");
			long sleepTime;
			if (!record())  {
				//执行失败后休眠一段时间再执行
				sleepTime = this.getMillis();
			} else {
				sleepTime = 1;
			}
			
			if (sleepTime > 0) {
				try {
					Thread.sleep(sleepTime);
				} catch (Exception ex) {}
			}
		} while(!isSystemShutdown);
	}

	protected Queue<T> newQueue() {
		queue = new ConcurrentLinkedQueue<T>();
		return queue;
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

		boolean add = queue.offer(tObj);
		synchronized (queue) {
			queue.notify();
		}

		return add;
	}

	/**
	 * 添加一批记录任务
	 * @param tList 需要记录的对象集合
	 */
	public boolean addRecordTaskList(List<T> tList) {
		//忽略空任务
		if(null == tList) {
			return false;
		}

		boolean add = queue.addAll(tList);
		synchronized (queue) {
			queue.notify();
		}
		return add;
	}

	/**
	 * 获取一个记录任务，同时将任务从队列中移除
	 * @param
	 * @return
	 */
	public T getRecordTask() {
		return queue.poll();
	}

	public static void main(String[] args) throws InterruptedException {
		AbstractFastRecord<String> f = new AbstractFastRecord<String>() {
			private List<String> l;

			@Override
			protected void clearCache() {
				if (null == l) {
					l = new ArrayList<String>();
				} else {
					l.clear();
				}
			}

			@Override
			protected boolean addTaskToCache(String tObj) {
				l.add(tObj);
				return true;
			}

			@Override
			protected boolean saveDataFromCache() throws Exception {
				for (int i = 0; i < l.size(); i++) {
					System.out.print(l.get(i) + '\t');
				}
				System.out.println();

				return false;
			}

			@Override
			protected void error(String info, Throwable throwable) {
				System.out.println(info);
				throwable.printStackTrace();
			}

		};
		for (int i = 0; i < 100; i++) {
			//Thread.sleep(2000);
			f.addRecordTask("take");
			Thread.sleep(1);
			f.addRecordTask(String.valueOf(i));
		}


		Thread.sleep(199999);
	}
}

