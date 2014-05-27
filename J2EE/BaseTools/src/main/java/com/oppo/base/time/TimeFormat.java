/**
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 * FileName:TimeFormat.java
 * ProjectName:OBaseTools
 * PackageName:com.oppo.base.time
 * Create Date:2011-4-7
 * History:
 *   ver	date	  author		desc	
 * ────────────────────────────────────────────────────────
 *   1.0	2011-4-7	  80036381		
 *
 * 
*/

package com.oppo.base.time;

import java.sql.Timestamp;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * ClassName:TimeFormat
 * Function: 时间格式处理工具类
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-4-7  上午09:34:04
 */
public class TimeFormat {
	/**
	 * 默认的时间表示格式
	 */
	public static final String DEFAULT_FORMAT = "yyyy-MM-dd HH:mm:ss";
	
	/**
	 * 默认的oracle时间格式
	 */
	public static final String ORACLE_FORMAT = "YYYY-MM-DD HH24:MM:SS";
	
	/**
	 * 获取当前时间的String格式
	 * @return
	 */
	public static String getCurrentTime() {
		return getTimeString(DEFAULT_FORMAT); //日期格式
	}
	
	/**
	 * 获取当前时间指定格式的字符串
	 * @param format 
	 * @return
	 */
	public static String getTimeString(String format) {
		return getTimeString(new Date(), format);
	}
	
	/**
	 * 获取指定日期的字符串
	 * @param timestamp	需要转换的时间戳
	 * @return
	 */
	public static String getTimeString(long timestamp) {
		return getTimeString(new Date(timestamp), DEFAULT_FORMAT);
	}
	
	/**
	 * 获取指定日期的字符串
	 * @param timestamp	需要转换的时间戳
	 * @param format 需要转换的格式
	 * @return
	 */
	public static String getTimeString(long timestamp, String format) {
		return getTimeString(new Date(timestamp), format);
	}
	
	/**
	 * 获取指定日期的字符串
	 * @param date	需要转换的日期
	 * @return
	 */
	public static String getTimeString(Date date) {
		return getTimeString(date, DEFAULT_FORMAT);
	}
	
	/**
	 * 获取指定格式指定日期的字符串
	 * @param date		需要转换的日期
	 * @param format	日期格式
	 * @return
	 */
	public static String getTimeString(Date date, String format) {
		SimpleDateFormat dateFm = new SimpleDateFormat(format); 
		return dateFm.format(date);
	}
	
	/**
	 * 从指定字符串解析出日期
	 * @param 日期字符串
	 * @return 成功则返回日期，失败返回null
	 */
	public static Date getDateFromString(String formatString) {
		return getDateFromString(formatString, DEFAULT_FORMAT);
	}
	
	/**
	 * 从指定字符串解析出日期
	 * @param 日期字符串
	 * @return 成功则返回日期，失败返回null
	 */
	public static Date getDateFromString(String formatString, String format) {
		SimpleDateFormat dateFm = new SimpleDateFormat(format); 
		
		try {
			return dateFm.parse(formatString);
		} catch (ParseException e) {
			return null;
		}
	}
	
	public static void main(String[] args) {
		System.out.println(Timestamp.valueOf("2011-04-13 23:10:19"));
	}
}

