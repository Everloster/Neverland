/**
 * TaskConfig.java
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

import java.io.Serializable;

/**
 * ClassName:TaskConfig <br>
 * Function: 任务配置 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-11-16  下午01:52:36
 */
public class TaskConfig implements Serializable {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	private String directory;	//数据文件路径
	private String filePrefix;	//数据文件名前缀
	private String handleClass;	//处理任务的类
	private String timePolicy;	//运行时间策略
	private long timeLimit;		//读取文件需要满足的最短未修改时间
	private String backupDir;	//备份目录
	
	/**
	 * 获取文件路径,如果有多个路径则以";"分开
	 * @return  the filePath
	 * @since   CodingExample Ver 1.0
	 */
	public String getDirectory() {
		return directory;
	}
	
	/**
	 * 设置文件路径,如果有多个路径则以";"分开
	 * @param   filePath    the filePath to set
	 * @since   CodingExample Ver 1.0
	 */
	public void setDirectory(String directory) {
		this.directory = directory;
	}
	
	/**
	 * 获取filePrefix
	 * @return  the filePrefix
	 * @since   CodingExample Ver 1.0
	 */
	public String getFilePrefix() {
		return filePrefix;
	}
	
	/**
	 * 设置文件前缀  
	 * @param   filePrefix    the filePrefix to set
	 * @since   CodingExample Ver 1.0
	 */
	public void setFilePrefix(String filePrefix) {
		this.filePrefix = filePrefix;
	}
	
	/**
	 * 获取handleClass
	 * @return  the handleClass
	 * @since   CodingExample Ver 1.0
	 */
	public String getHandleClass() {
		return handleClass;
	}
	
	/**
	 * 设置handleClass  
	 * @param   handleClass    the handleClass to set
	 * @since   CodingExample Ver 1.0
	 */
	public void setHandleClass(String handleClass) {
		this.handleClass = handleClass;
	}
	
	/**
	 * 获取运行策略
	 * @return  the timePolicy
	 * @since   Ver 1.0
	 */
	public String getTimePolicy() {
		return timePolicy;
	}
	
	/**
	 * 设置运行策略 
	 * @param   timePolicy    时间策略
	 * @since   Ver 1.0
	 */
	public void setTimePolicy(String timePolicy) {
		this.timePolicy = timePolicy;
	}

	/**
	 * 获取读取文件需要满足的最短未修改时间,以毫秒为单位
	 * @return  sth.
	 * @since   Ver 1.0
	 */
	public long getTimeLimit() {
		return timeLimit;
	}

	/**
	 * 设置读取文件需要满足的最短未修改时间,以毫秒为单位
	 * @param   timeLimit	sth.   
	 * @since   Ver 1.0
	 */
	public void setTimeLimit(long timeLimit) {
		this.timeLimit = timeLimit;
	}

	/**
	 * 获取备份目录
	 * @return  the backupDir
	 * @since   Ver 1.0
	 */
	public String getBackupDir() {
		return backupDir;
	}

	/**
	 * 设置备份目录
	 * @param   backupDir    
	 * @since   Ver 1.0
	 */
	public void setBackupDir(String backupDir) {
		this.backupDir = backupDir;
	}
}

