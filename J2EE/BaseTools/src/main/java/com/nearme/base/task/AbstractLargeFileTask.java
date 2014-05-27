/**
 * AbstractLargeFileTask.java
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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;
import com.oppo.base.file.FileOperate;

/**
 * ClassName:AbstractLargeFileTask <br>
 * Function: 文件处理任务，适合处理大文件 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-11-16  下午02:12:26
 */
public abstract class AbstractLargeFileTask extends AbstractTask {

	@Override
	public void executeTask() {
		TaskConfig tc = this.getTaskConfig();			//任务配置
		long currentTime = System.currentTimeMillis(); 	//当前时间

		//判断目录是否存在
		String[] directories = StringUtil.split(tc.getDirectory(), ';');
		int length = directories.length;
		for(int i = 0; i < length; i++) {
			File dir = new File(directories[i]);
			if(!dir.isDirectory()) {
				getLog().error(dir.getAbsolutePath() + " is not a directory.");
				return;
			}
					
			handleDirectory(dir, tc, currentTime, false);
		}
	}
	
	/**
	 * 处理目录日志
	 * @param dir 日志目录
	 * @return
	 */
	private void handleDirectory(File dir, TaskConfig tc, long currentTime, boolean isDelete) {
		//遍历目录
		File[] subFiles = dir.listFiles();
		int fileCount = subFiles.length;
		//不存在文件则直接返回
		if(fileCount == 0) {
			if(isDelete) {
				dir.delete();
			}
			return;
		}

		String filePrefix = tc.getFilePrefix().toLowerCase();
		for (int i = 0; i < fileCount; i++) {
			File subFile = subFiles[i];
			if (subFile.isDirectory()) { // 忽略目录
				handleDirectory(subFile, tc, currentTime, true);
			} else {
				//判断前缀是否符合
				if(!StringUtil.isNullOrEmpty(filePrefix)
						&& !subFile.getName().toLowerCase().startsWith(filePrefix)) {
					continue;
				}
				
				// 如果超过最短未修改时间则对文件进行处理
				long lastModify = subFile.lastModified();
				if (currentTime - lastModify > tc.getTimeLimit()) {
					if (!subFile.canRead()) {
						// 记录不可读文件
						getLog().warn(subFile.getAbsolutePath() + " can not be read.");
						continue;
					} else {
						if (handleFile(subFile)) {
							clearFile(subFile);
						}
					}
				}
			}
		}
	}
	
	
	/**
	 * 处理单个文件
	 * @param handleFile 需要处理的文件
	 * @return
	 */
	protected boolean handleFile(File handleFile) {
		BufferedReader bufReader = null;
		try {
			handleFileBegin();
			
			bufReader = new BufferedReader(new FileReader(handleFile));
			String line;	//行数据
			int handleCount = 0;
			while((line = bufReader.readLine()) != null) {
				handleCount += handleFileLine(new String(line.getBytes(), OConstants.DEFAULT_ENCODING));
			}
			
			//防止有未执行的数据，再执行一次
			handleCount += handleFileEnd();
			
			return handleCount > 0;
		} catch(Exception e) {
			getLog().error(handleFile.getAbsolutePath() + "执行出错.", e);
			return false;
		} finally {
			FileOperate.close(bufReader);
		}
	}

	/**
	 * 清理处理过的文件,如果有备份目录则备份，否则直接删除
	 * @param handleFile 要处理的文件
	 * @return
	 */
	protected void clearFile(File handleFile) {
		//备份文件
		String backupDir = this.getTaskConfig().getBackupDir();
		if(!StringUtil.isNullOrEmpty(backupDir)) {
			String originalDir = new File(this.getTaskConfig().getDirectory()).getAbsolutePath();
			String suffix = handleFile.getAbsolutePath().substring(originalDir.length());
			File newFile = new File(backupDir, suffix);
			int i = 0;
			while(newFile.exists() && i++ < 10) {
				newFile = new File(backupDir, suffix + "_" + i);
			}
			FileOperate.createDir(newFile, true);
			if(!handleFile.renameTo(newFile)) {
				//移动失败则直接删除
				handleFile.delete();
			}
		} else {
			//删除文件
			boolean isSucc = handleFile.delete();
			if(!isSucc) { 
				getLog().warn("删除文件" + handleFile.getAbsolutePath() + "失败.");
			}
		}
	}

	/**
	 * 处理文件之前进行的操作 
	 * @return
	 */
	protected abstract void handleFileBegin() throws Exception;
	
	/**
	 * 
	 * 处理单行数据
	 * @param line 行数据
	 * @return
	 */
	protected abstract int handleFileLine(String line) throws Exception;
	
	/**
	 * 
	 * 文件处理末尾需要处理的内容
	 * @return
	 */
	protected abstract int handleFileEnd() throws Exception;
}

