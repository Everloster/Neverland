/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年3月12日
 */
package org.jabe.neverland.download.core;

/**
 * 
 * @Author	LaiLong
 * @Since	2014年3月12日
 */
public interface DownloadRegister {

	public void registerListener(DownloadListener downloadInterface);
	
	public void removeListenter(DownloadListener downloadInterface);

}
