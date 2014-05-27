/**
 * AbstractTask.java
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

import java.io.File;
import java.io.InputStream;
import java.util.Date;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.oppo.base.common.NumericUtil;
import com.oppo.base.time.TimerPolicy;
import com.oppo.base.xml.XmlConfigReader;
import com.oppo.base.xml.XmlSimpleMapParse;

/**
 * ClassName:AbstractTask <br>
 * Function: 抽象任务实现 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-11-16  下午01:56:41
 */
public abstract class AbstractTask implements ITask {
	/**
	 * 任务配置
	 */
	private TaskConfig taskConfig;
	
	//日志记录类
	private Logger log;
	
	public AbstractTask() {
		//初始化日志记录类
		log = LoggerFactory.getLogger(this.getClass());
	}
	
	/**
	 * 检查是否计划可执行
	 * @see com.nearme.base.task.ITask#canExecute(java.util.Date)
	 */
	@Override
	public boolean canExecute(Date date) {
		return TimerPolicy.canStart(date.getTime(), getTaskConfig().getTimePolicy());
	}

	/**
	 * 初始化配置
	 * @param map
	 * @see com.nearme.base.task.ITask#initial(java.util.Map)
	 */
	@Override
	public void initial(Map<String, String> map) {
		taskConfig = initTaskConfig();
		//初始化基本任务配置
		taskConfig.setDirectory(map.get(ITask.FILE_PATH_SEC));
		taskConfig.setBackupDir(map.get(ITask.FILE_BACKUP_SEC));
		taskConfig.setFilePrefix(map.get(ITask.FILE_PREFIX_SEC));
		taskConfig.setTimePolicy(map.get(ITask.TIME_POLICY_SEC));
		//设置最短未修改时间
		long timeLimit = NumericUtil.parseLong(map.get(ITask.TIME_LIMIT_SEC), ITask.DEFAULT_TIME_LIMIT);
		taskConfig.setTimeLimit(timeLimit);
	}
	
	/**
	 * 通过inputStream中的xml初始化配置
	 * @param inputStream
	 */
	public void initial(InputStream inputStream) {
		XmlSimpleMapParse xsp = new XmlSimpleMapParse();
		xsp.readFromFile(inputStream);
		
		initial(xsp.getXmlMap());
	}
	
	/**
	 * 通过文件中的xml初始化配置
	 * @param file
	 */
	public void initial(File file) {
		XmlConfigReader configReader = XmlConfigReader.getXmlConfig(file);
		
		initial(configReader.getConfigMap());
	}
	
	@Override
	public void run() {
		//尝试加锁，防止任务未执行完成就开始下一次的执行
		ReentrantLock lock = (ReentrantLock)getLock();
		if(null != lock) {
			if(!lock.tryLock()) {
				return;
			}
		}
		
		try {
			taskFlow();
		} catch(Exception ex) {
			log.error("任务执行出错", ex);
		} finally {
			//释放资源
			release();
			
			if(null != lock) {
				lock.unlock();
			}
		}
	}
	
	/**
	 * 基本任务执行流程： 任务前初始化 -> 执行任务 -> 任务后清理 -> 释放相关资源
	 * @param 
	 * @return
	 * @throws Exception 
	 */
	protected void taskFlow() throws Exception {
		//任务开始前的初始化
		doBeforeTask();
			
		//执行任务
		executeTask();
			
		//任务结束后的处理
		doAfterTask();
	}
	
	/**
	 * 获取进程锁，防止任务上次未执行完就再次执行，建议用static final修饰
	 * @param 
	 * @return 进程的锁，返回空表示不锁
	 */
	protected abstract ReentrantLock getLock();
	
	/**
	 * 任务开始时的处理(一般为工具类的初始化)
	 * @param 
	 * @return
	 */
	protected abstract void doBeforeTask() throws Exception;
	
	/**
	 * 执行任务
	 * @param 
	 * @return
	 */
	protected abstract void executeTask() throws Exception;
	
	/**
	 * 任务结束后的处理
	 * @param 
	 * @return
	 */
	protected abstract void doAfterTask() throws Exception;
	
	/**
	 * 释放资源
	 */
	protected abstract void release();
	
	/**
	 * 获取初始化后的任务
	 * @return
	 */
	public TaskConfig initTaskConfig() {
		return new TaskConfig();
	}

	/**
	 * 获取taskConfig
	 * @return  sth.
	 * @since   Ver 1.0
	 */
	public TaskConfig getTaskConfig() {
		return taskConfig;
	}

	/**
	 * 设置taskConfig  
	 * @param   taskConfig	sth.   
	 * @since   Ver 1.0
	 */
	public void setTaskConfig(TaskConfig taskConfig) {
		this.taskConfig = taskConfig;
	}
	
	/**
	 * 获取log
	 * @return  the log
	 * @since   Ver 1.0
	 */
	protected Logger getLog() {
		return log;
	}
}

