/**
 * AbstractFileRunnableRecord.java
 * com.oppo.base.record
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-9-30 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.record;

import com.oppo.base.file.FileOperate;

/**
 * ClassName:AbstractFileRunnableRecord
 * Function: 文件保存操作
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-30  上午08:41:46
 */
public abstract class AbstractFileRunnableRecord<T> extends AbstractRunnableRecord<T> {
	protected StringBuilder textBuilder;	//存储文件内容的builder
	
	@Override
	protected void clearCache() {
		if(null != textBuilder) {
			textBuilder.setLength(0);
		} else {
			textBuilder = new StringBuilder();
		}
	}

	@Override
	protected boolean addTaskToCache(T tObj) {
		textBuilder.append(getString(tObj)).append(getLineDelimiter());
		return true;
	}

	@Override
	protected boolean saveDataFromCache() throws Exception {
		return FileOperate.appendContentToFile(getSavePath(), textBuilder.toString());
	}

	/**
	 * 获取行分隔符
	 * @param 
	 * @return
	 */
	protected String getLineDelimiter() {
		return "\r\n";
	}
	
	/**
	 * 获取保存的路径
	 * Function Description here
	 * @param 
	 * @return
	 */
	protected abstract String getSavePath();

	/**
	 * 通过对象获取其String内容，此tObj不为null,不需对其进行null判断
	 * @param 
	 * @return
	 */
	protected abstract String getString(T tObj);
}

