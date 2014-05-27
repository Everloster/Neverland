/**
 * DirectoryOperate.java
 * com.oppo.base.file
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-21 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.file;

import java.io.File;

/**
 * ClassName:DirectoryOperate
 * Function: 目录相关操作
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-21  上午11:45:47
 */
public class DirectoryOperate {
	/**
	 * 删除指定路径的文件夹
	 * @param 
	 * @return
	 */
	public static boolean deleteDirectory(String dirPath, boolean deleteSubFiles) {
		return deleteDirectory(new File(dirPath), deleteSubFiles);
	}
	
	/**
	 * 删除指定路径的文件夹
	 * @param 
	 * @return
	 */
	public static boolean deleteDirectory(File dir, boolean deleteSubFiles) {
		if(!dir.exists() || !dir.isDirectory()) {
			return false;
		}
		
		//获取所有子文件(夹)
		File[] subFiles = dir.listFiles();
		int subFilesLen = subFiles.length;
		//如果不能删除子文件(夹)且包含子文件夹则删除失败
		if(!deleteSubFiles && subFilesLen > 0) {
			return false;
		}
		
		boolean isDelete = true;
		//开始删除子文件夹
		for(int i = 0; i < subFilesLen; i++) {
			File file = subFiles[i];
			if(file.isFile()) {
				if(!file.delete()) {
					isDelete = false;
				}
			} else if(file.isDirectory()) {
				if(!deleteDirectory(file, deleteSubFiles)) {
					isDelete = false;
				}
			}
		}
		
		//删除文件夹本身
		if(isDelete) {
			isDelete = dir.delete();
		}
		return isDelete;
	}
}

