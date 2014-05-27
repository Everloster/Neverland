/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年4月8日
 */
package org.jabe.neverland.download.util;

import java.text.DecimalFormat;
import java.text.NumberFormat;

/**
 * 
 * @Author	LaiLong
 * @Since	2014年4月8日
 */
public class SpeedDetectUtil {

	/**
	 * 
	 */
	public SpeedDetectUtil() {
	}

	
	public static String getSpeedString(long castTime, long bytes) {
		double castSecond = (double)castTime / (double) 1000;
		return parseSpeedString((long)((double)bytes/castSecond)/1024);
	}
	
	private static final NumberFormat SIZE_FORMAT_M = new DecimalFormat("###.##M/s");
	private static final NumberFormat SIZE_FORMAT_K = new DecimalFormat("####K/s");
	
	private static String parseSpeedString(long size) {
		if (size > 1024) {
			return SIZE_FORMAT_M.format(((float) size) / 1024);
		}
		return SIZE_FORMAT_K.format(size);
	}
}
