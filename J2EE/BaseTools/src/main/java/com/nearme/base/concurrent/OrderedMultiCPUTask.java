/**
 * OrderedMultiCPUTask.java
 * com.nearme.base.concurrent
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-5-11 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.concurrent;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.FutureTask;

/**
 * ClassName:OrderedMultiCPUTask <br>
 * Function: 多任务按顺序获取结果 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-5-11  下午03:24:44
 */
public class OrderedMultiCPUTask extends MultiCPUTask {
	/**
	 * 
	 * @param 
	 * @return
	 */
	protected <T, D> List<D> doRealMultiHandle(final List<T> dataList, 
			ExecutorService currentExecutor,
			int threadCount,
			final IMultiCPUHandler<T, D> handler) throws InterruptedException, ExecutionException {
		int listSize = dataList.size();
		//算出每个任务负责的记录数
		int taskHandleCount = getDevideResult(listSize, threadCount);
		
		//创建一个处理任务存储列表
		FutureTask<List<D>>[] taskArray = new FutureTask[threadCount];
		//任务拆分处理
		for(int i = 0; i < threadCount; i++) {
			//得到任务处理范围
			final int startIndex = i * taskHandleCount;
			int endIndex = (i + 1) * taskHandleCount - 1;
			if(endIndex >= listSize) {
				endIndex = listSize - 1;
			}
			final int realEndIndex = endIndex;
			
			//创建任务
			FutureTask<List<D>> task = new FutureTask<List<D>>(new Callable<List<D>>() {
				@Override
				public List<D> call() throws Exception {
					return handler.handle(dataList, startIndex, realEndIndex);
				}
			});
			
			taskArray[i] = task;
			//提交任务
			currentExecutor.submit(task);
		}
		
		//处理任务
		List<D> results = new ArrayList<D>();
		for(FutureTask<List<D>> task : taskArray) {
			List<D> result = task.get();
			if(null != result) {
				results.addAll(result);
				result.clear();
				result = null;
			}
		}
		
		return results;
	}
}

