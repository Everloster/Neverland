/**
 * IRecord.java
 * com.oppo.base.record
 *
 * Function：  
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-21 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.record;
/**
 * ClassName:IRecord
 * Function: 信息记录
 * 调用过程应该为：
 * 1.添加任务 addRecordTask
 * 2.判断是否可以进行日志记录 canRecord
 * 3.执行记录过程 record
 * 
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-4-6  下午08:12:24
 */
public interface IRecord<T> {
	/**
	 * 
	 * 添加需要记录的信息任务
	 * @param T 记录对应的实体
	 * @return 是否添加成功
	 */
	boolean addRecordTask(T tObj);
	
	/**
	 * 
	 * 是否可以进行记录
	 * @return
	 */
	boolean canRecord();
	
	/**
	 * 记录信息
	 * @return 是否记录成功
	 */
	boolean record();
}

