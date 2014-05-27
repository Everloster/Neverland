/**
 * ConcurrentTest.java
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

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;

/**
 * ClassName:ConcurrentTest
 * Function: 并发测试
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-10-21  上午09:21:50
 */
public abstract class ConcurrentTest {
	protected String testId;
    protected CyclicBarrier barrier;
    protected long count;
    protected int threadNum;
    protected ExecutorService executor;

    public ConcurrentTest(String testId) {
    	this.testId = testId;
    }

    public ConcurrentTest(String testId, CyclicBarrier barrier, long count, int threadNum,
            ExecutorService executor) {
        this.testId = testId;
        this.barrier = barrier;
        this.count = count;
        this.threadNum = threadNum;
        this.executor = executor;
    }

    /**
     * 启动测试线程
     * @param
     * @return
     */
    public long getTestDuration() {
        long start = System.currentTimeMillis();

        for (int j = 0; j < threadNum; j++) {
            executor.execute(new Runnable() {
                @Override
                public void run() {
                    for (int i = 0; i < count; i++) {
                        test();
                    }

                    try {
                        barrier.await();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    } catch (BrokenBarrierException e) {
                        e.printStackTrace();
                    }
                }
            });
        }

        try {
            barrier.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (BrokenBarrierException e) {
            e.printStackTrace();
        }

        // 所有线程执行完成之后，才会跑到这一步
        return System.currentTimeMillis() - start;
    }

    /**
     * 被测试的内容
     * @param
     * @return
     */
    protected abstract void test();

	/**
	 * 获取testId
	 * @return  the testId
	 * @since   Ver 1.0
	 */
	public String getTestId() {
		return testId;
	}

	/**
	 * 设置testId
	 * @param   testId
	 * @since   Ver 1.0
	 */
	public void setTestId(String testId) {
		this.testId = testId;
	}

	/**
	 * 获取barrier
	 * @return  the barrier
	 * @since   Ver 1.0
	 */
	public CyclicBarrier getBarrier() {
		return barrier;
	}

	/**
	 * 设置barrier
	 * @param   barrier
	 * @since   Ver 1.0
	 */
	public void setBarrier(CyclicBarrier barrier) {
		this.barrier = barrier;
	}

	/**
	 * 获取count
	 * @return  the count
	 * @since   Ver 1.0
	 */
	public long getCount() {
		return count;
	}

	/**
	 * 设置count
	 * @param   count
	 * @since   Ver 1.0
	 */
	public void setCount(long count) {
		this.count = count;
	}

	/**
	 * 获取threadNum
	 * @return  the threadNum
	 * @since   Ver 1.0
	 */
	public int getThreadNum() {
		return threadNum;
	}

	/**
	 * 设置threadNum
	 * @param   threadNum
	 * @since   Ver 1.0
	 */
	public void setThreadNum(int threadNum) {
		this.threadNum = threadNum;
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

