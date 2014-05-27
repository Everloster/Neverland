/**
 * FileCache2.java
 * com.nearme.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-30 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache;

import java.io.File;

import com.nearme.base.cache.selector.CacheSelector;
import com.oppo.base.cache.FileCacheObject;
import com.oppo.base.cache.util.FileCacheMonitor;
import com.oppo.base.file.FileOperate;
import com.oppo.base.security.MD5;

/**
 * ClassName:FileCache2 <br>
 * Function: 文件缓存 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-30  上午10:43:25
 */
public class FileCache2 extends AbstractCache2 {
	private String directory;
	private int[] directoryLevel;
	
	public FileCache2() {
		super();
	}
	
	public FileCache2(String directory) {
		super();
		
		this.directory = directory;
	}

	@Override
	public boolean expireSupported() {
		return true;
	}

	@Override
	public Object getCache(CacheSelector cacheSelector) throws Exception {
		File file = new File(this.getRealFilePath(cacheSelector.getCacheKey()));
		if(!file.exists()) {
			return null;
		}
		
		Object fileObj = FileOperate.getFileObject(file);//文件中保存的对象
		Object fileStorageObject = null;	//文件中保存的对象
		if(null != fileObj && fileObj instanceof FileCacheObject) {
			FileCacheObject fcObject = (FileCacheObject)fileObj;
			//缓存未过期才能获取对象
			if(!fcObject.isCacheExpire()) {
				fileStorageObject = fcObject.getStorageObject();
			}
		}
		
		//如果不存在或已过期则删除
		if(null == fileStorageObject) {
			file.delete();
		}
		
		return fileStorageObject;
	}

	@Override
	public boolean setCache(CacheSelector cacheSelector, Object cacheValue)
			throws Exception {
		long timeout = cacheSelector.getTimeout();
		String filePath = this.getRealFilePath(cacheSelector.getCacheKey());
		boolean isSave = FileOperate.saveObjectToFile(filePath, new FileCacheObject(timeout, cacheValue));
		if(isSave && timeout > 0) {
			FileCacheMonitor.getInstance().setFileCacheTime(filePath, timeout);
		}
		
		return isSave;
	}

	@Override
	public int insertCache(CacheSelector cacheSelector) throws Exception {
		//如果只有这一层则在该层中插入
		if(null == this.getNextCache()) {
			return this.setCache(cacheSelector, cacheSelector.getSelectorValue()) ? 1 : 0;
		} else {
			return 0;
		}
	}

	@Override
	public int updateCache(CacheSelector cacheSelector) throws Exception {
		//如果只有这一层则在该层中更新
		if(null == this.getNextCache()) {
			return this.setCache(cacheSelector, cacheSelector.getSelectorValue()) ? 1 : 0;
		} else {
			return 0;
		}
	}

	@Override
	public int deleteCache(CacheSelector cacheSelector) throws Exception {
		String cacheId = cacheSelector.getCacheKey();
		String filePath = this.getRealFilePath(cacheId);
		
		File f = new File(filePath);
		//文件缓存不存在则直接返回已删除
		if(!f.exists()) {
			return 1;
		} else {
			return f.delete() ? 1 : 0;
		}
	}

	@Override
	public void releaseResource() {
	}
	
	/**
	 * 获取cache所存的文件路径
	 * @param cacheId
	 * @return
	 */
	protected String getRealFilePath(String cacheId) {
		StringBuilder sb = new StringBuilder(directory);
		if(!directory.endsWith(File.separator)){
			sb.append(File.separatorChar); 
		}
			
		cacheId = cacheId.toLowerCase();
		int hashCode = Math.abs(cacheId.hashCode());
		int[] levels = getDirectoryLevel();
		for(int i = 0; i < levels.length; i++) {
			sb.append(hashCode % levels[i]).append(File.separatorChar);
		}

		sb.append(MD5.md5(cacheId));
		return sb.toString();
	}
	
	public String getDirectory() {
		return directory;
	}

	/**
	 * 设置文件缓存目录
	 * @param directory
	 */
	public void setDirectory(String directory) {
		this.directory = directory;
	}

	/**
	 * 获取目录层
	 * @return  the directoryLevel
	 * @since   Ver 1.0
	 */
	public int[] getDirectoryLevel() {
		//默认为2层，第一层50个文件夹，第二层100个文件夹
		if(null == directoryLevel) {
			directoryLevel = new int[] { 50, 100 };
		}
		
		return directoryLevel;
	}

	/**
	 * 设置目录层
	 * @param   directoryLevel    
	 * @since   Ver 1.0
	 */
	public void setDirectoryLevel(int[] directoryLevel) {
		this.directoryLevel = directoryLevel;
	}
}

