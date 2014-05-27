/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年4月8日
 */
package org.jabe.neverland.download.core;

/**
 * 
 * @Author	LaiLong
 * @Since	2014年4月8日
 */
public interface DownloadSpeedListener extends DownloadListener {
	
	public void onSpeedNotify(String message, String packageName, long downloadBytes, long castTime);
}
