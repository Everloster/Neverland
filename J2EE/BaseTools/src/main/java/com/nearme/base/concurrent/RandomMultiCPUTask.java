/**
 * RandomMultiCPUTask.java
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
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;

/**
 * ClassName:RandomMultiCPUTask <br>
 * Function: 多任务随机获取结果，如果结果不需要保持顺序，则推荐使用此类 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-5-11  下午03:30:13
 */
public class RandomMultiCPUTask extends MultiCPUTask {

	protected <T, D> List<D> doRealMultiHandle(final List<T> dataList, 
			ExecutorService currentExecutor,
			int threadCount,
			final IMultiCPUHandler<T, D> handler) throws InterruptedException, ExecutionException {
		int listSize = dataList.size();
		//算出每个任务负责的记录数
		int taskHandleCount = getDevideResult(listSize, threadCount);
		
		CompletionService<List<D>> completionService = new ExecutorCompletionService<List<D>>(currentExecutor); 
		
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
			Callable<List<D>> task = new Callable<List<D>>() {
				@Override
				public List<D> call() throws Exception {
					return handler.handle(dataList, startIndex, realEndIndex);
				}
			};
			
			//提交任务
			completionService.submit(task);
		}
		
		//处理任务
		List<D> results = new ArrayList<D>();
		for (int i = 0; i < threadCount; i++) {              
            List<D> result = completionService.take().get();  
            if(null != result) {
    			results.addAll(result);
    			result.clear();
    			result = null;
    		}
        }
		
		return results;
	}
}

