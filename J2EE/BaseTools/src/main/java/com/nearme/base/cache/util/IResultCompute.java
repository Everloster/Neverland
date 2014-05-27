/**
 * IResultCompute.java
 * com.nearme.base.cache.util
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-30 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache.util;


/**
 * ClassName:IResultCompute <br>
 * Function: 结果计算 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-30  下午06:58:10
 */
public interface IResultCompute<T> {
	/**
	 * 结果计算
	 * @param 
	 * @return
	 */
	public T compute();
	
	/**
	 * 获得数据的最后加载时间
	 * @param 
	 * @return
	 */
	public long getLastLoadTime();
}

