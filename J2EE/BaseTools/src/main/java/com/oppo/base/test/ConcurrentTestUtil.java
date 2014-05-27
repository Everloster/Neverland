/**
 * ConcurrentTestUtil.java
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

import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * ClassName:ConcurrentTestUtil
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-21  下午02:00:13
 */
public class ConcurrentTestUtil {
	 
	/**
	 * 指定线程数、执行次数对测试对象进行时间对比
	 * @param count 执行次数
	 * @param threadNum 线程数
	 * @param testObjects 进行对比测试的对象
	 * @return
	 */
    public static void compare(int count, int threadNum, ConcurrentTest...testObjects) { 
    	
    	ExecutorService executor = Executors.newFixedThreadPool(threadNum);
    	
    	final CyclicBarrier barrier = new CyclicBarrier(threadNum + 1,  
                new Runnable() {  
                    @Override  
                    public void run() {
                    }  
                }); 
		
		System.out.println("===============-start-===============");  
        System.out.println("count = " + count + "\t" + "Thread Count = " + threadNum); 
        
        int length = testObjects.length;
        for(int i = 0; i < length; i++) {
        	ConcurrentTest testObject = testObjects[i];
        	testObject.setBarrier(barrier);
        	testObject.setCount(count);
        	testObject.setExecutor(executor);
        	testObject.setThreadNum(threadNum);
        	long testDuration = testObject.getTestDuration();
        	System.out.println(testObject.getTestId() + "\t:" + testDuration);
        }
		
		System.out.println("===============--end--==============="); 
		
		executor.shutdownNow();
    }
}

