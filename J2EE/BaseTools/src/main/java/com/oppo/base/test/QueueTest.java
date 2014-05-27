/**
 * QueueTest.java
 * com.oppo.base.test
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-21 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.test;

import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.locks.ReentrantLock;

/**
 * ClassName:QueueTest
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-21  上午11:35:32
 */
public class QueueTest {
	public static void main(String[] args) {
		for (int i = 1; i < 5; i++) {  
			ConcurrentTestUtil.compare(1000 * i, 100 * i, 
					new LinkedQueueSync(),	//速度排名3
					new LinkedQueueLock(),	//速度排名1
					//new LinkedQueueLock(),//速度排名，不调用size时最快
					new BlockQueue());  	//速度排名2,加入poll后比较慢
        }
	}
	
	public static void operateQueue(Queue<Object> queue) {
		queue.add(1);
		if(queue.size() > 1000) {
			queue.clear();
		}
		queue.poll();
	}
}

class LinkedQueueSync extends ConcurrentTest {
	private static final Object COMMON_LOCK = new Object();
	private LinkedList<Object> l = new LinkedList<Object>();
	
	public LinkedQueueSync() {
		super("LinkedListSync");
	}
	
	@Override
	protected void test() {
		synchronized(COMMON_LOCK) {
			QueueTest.operateQueue(l);
		}
	}
}

class LinkedQueueLock extends ConcurrentTest {
	private static final ReentrantLock lock = new ReentrantLock();
	private LinkedList<Object> l = new LinkedList<Object>();
	
	public LinkedQueueLock() {
		super("LinkedListLock");
	}

	@Override
	protected void test() {
		lock.lock();
		
		try {
			QueueTest.operateQueue(l);
		} finally {
			lock.unlock();
		}
	}
}

class ConcurrentQueue extends ConcurrentTest {
	private ConcurrentLinkedQueue<Object> l = new ConcurrentLinkedQueue<Object>();
	public ConcurrentQueue() {
		super("ConcurrentQueue");
	}

	@Override
	protected void test() {
		QueueTest.operateQueue(l);
	}
}

class BlockQueue extends ConcurrentTest {
	private LinkedBlockingQueue<Object> l = new LinkedBlockingQueue<Object>();
	public BlockQueue() {
		super("BlockingQueue");
	}

	@Override
	protected void test() {
		QueueTest.operateQueue(l);
	}
}
