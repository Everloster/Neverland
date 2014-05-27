/**
 * IMultiCPUHandler.java
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

/**
 * ClassName:IMultiCPUHandler <br>
 * Function: 多cpu任务处理的处理类 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-5-9  下午05:37:04
 */
public interface IMultiCPUHandler<T, D> {
	/**
	 * 处理指定范围的数据，并返回处理结果
	 * @param 
	 * @return
	 */
	List<D> handle(List<T> dataList, int startIndex, int endIndex);
}

