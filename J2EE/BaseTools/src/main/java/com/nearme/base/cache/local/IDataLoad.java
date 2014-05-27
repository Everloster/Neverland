/**
 * IDataLoad.java
 * com.nearme.base.cache.local
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-28 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache.local;
/**
 * ClassName:IDataLoad <br>
 * Function: 定期从持久层中获取数据进行处理，处理完成后修改最后加载时间，使上层缓存可获取到数据的最新状态。<br>
 *  同时可接收外部的数据刷新通知。<br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-28  下午03:18:53
 */
public interface IDataLoad extends Runnable {
	/**
	 * 初始化数据，一般是从轻量的持久层中获取数据，使系统启动过程更短
	 * @param flag 数据加载时提供的标记
	 * @return 是否初始化成功
	 */
	boolean initData(Object flag);
	
	/**
	 * 从持久层中获取数据，一般由对象内部调用
	 * @param flag 数据加载时提供的标记
	 * @return 是否加载成功
	 */
	boolean loadData(Object flag);
	
	/**
	 * 数据刷新，一般由外部对象调用
	 * @param flag 刷新时提供的标记
	 * @return 是否刷新成功
	 */
	boolean flushData(Object flag);
	
	/**
	 * 获取数据的最后加载时间，外部对象可根据此方法判断是否需要重新加载该层缓存中的数据
	 * @param 
	 * @return
	 */
	long getLastLoadTime(Object flag);
}

