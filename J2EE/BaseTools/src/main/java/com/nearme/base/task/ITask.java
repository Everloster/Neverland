/**
 * ITask.java
 * com.nearme.base.task
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-11-16 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.task;

import java.util.Date;
import java.util.Map;

/**
 * ClassName:ITask <br>
 * Function: 任务接口, 线程执行 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-11-16  下午01:51:28
 */
public interface ITask extends Runnable {

	/**
	 * 处理任务的类名
	 */
	public static final String HANDLE_CLASS_SEC = "handleClass";
	
	/**
	 * 任务需要处理的文件夹
	 */
	public static final String FILE_PATH_SEC = "directory";
	
	/**
	 * 任务文件备份目录
	 */
	public static final String FILE_BACKUP_SEC = "backupDir";
	
	/**
	 * 任务需要处理的文件前缀
	 */
	public static final String FILE_PREFIX_SEC = "filePrefix";
	
	/**
	 * 任务执行的时间策略
	 */
	public static final String TIME_POLICY_SEC = "timePolicy";
	
	/**
	 * 读取文件需要满足的最短未修改时间
	 */
	public static final String TIME_LIMIT_SEC = "timeLimit";
	
	/**
	 * 默认最短未修改时间
	 */
	public static final long DEFAULT_TIME_LIMIT = 1800000;
	
	/**
	 * 是否可以执行任务
	 * @param 
	 * @return
	 */
	boolean canExecute(Date date);
	
	/**
	 * 利用指定配置进行初始化
	 * @param 
	 * @return
	 */
	void initial(Map<String, String> map);
}

