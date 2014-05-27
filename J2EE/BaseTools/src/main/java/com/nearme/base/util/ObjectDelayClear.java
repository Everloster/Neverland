/**
 * ObjectDelayClear.java
 * com.nearme.base.util
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-11-2 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.util;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * ClassName:ObjectDelayClear <br>
 * Function: 对象延迟清理，防止用到此对象的的任务出现错误 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-11-2  上午11:46:50
 */
public final class ObjectDelayClear {
	/**
	 * 默认延迟时间，已有数据将在该延迟后清除
	 */
	public static long DEFAULT_DELAY = 5;	
	
	/**
	 * 默认延迟时间的单位
	 */
	public static TimeUnit DEFAULT_UNIT = TimeUnit.SECONDS;
	
	private static final ScheduledThreadPoolExecutor scheduledExecutor = new ScheduledThreadPoolExecutor(3);
	
	/**
	 * 清理List,默认30秒后清理
	 * @param 
	 * @return
	 */
	public static void clear(List<?> list) {
		clear(list, DEFAULT_DELAY, DEFAULT_UNIT);
	}
	
	/**
	 * 指定时间清理List
	 * @param 
	 * @return
	 */
	public static void clear(final List<?> list, long delay, TimeUnit unit) {
		if(null == list) {
			return;
		}
		
		scheduledExecutor.schedule(new Runnable() {
			@Override
			public void run() {
				list.clear();
			}
		}, delay, unit);
	}
	
	/**
	 * 清理Map,默认30秒后清理
	 * @param 
	 * @return
	 */
	public static void clear(Map<?, ?> map) {
		clear(map, DEFAULT_DELAY, DEFAULT_UNIT);
	}
	
	/**
	 * 指定时间清理map
	 * @param 
	 * @return
	 */
	public static void clear(final Map<?, ?> map, long delay, TimeUnit unit) {
		if(null == map) {
			return;
		}
		
		scheduledExecutor.schedule(new Runnable() {
			@Override
			public void run() {
				map.clear();
			}
		}, delay, unit);
	}
}

