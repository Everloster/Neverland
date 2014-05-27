package com.oppo.base.common;

/**
 * 数字相关操作
 * @author 80036381
 * 2011-1-7
 */
public class NumericUtil {
	/**
	 * 解析字符串为int，解析失败则返回默认值
	 * @param key
	 * @param defaultValue
	 * @return
	 */
	public static int parseInt(String key, int defaultValue) {
		int value = defaultValue;
		if(!StringUtil.isNullOrEmpty(key)){
			try{
				value = Integer.parseInt(key);
			} catch(Exception e) {
			}
		}
		
		return value;
	}
	
	/**
	 * 解析字符串为Long，解析失败则返回默认值
	 * @param key
	 * @param defaultValue
	 * @return
	 */
	public static long parseLong(String key, long defaultValue) {
		long value = defaultValue;
		if(!StringUtil.isNullOrEmpty(key)){
			try{
				value = Long.parseLong(key);
			} catch(Exception e) {
			}
		}
		
		return value;
	}
	
	/**
	 * 解析字符串为Float，解析失败则返回默认值
	 * @param key
	 * @param defaultValue
	 * @return
	 */
	public static float parseFloat(String key, float defaultValue) {
		float value = defaultValue;
		if(!StringUtil.isNullOrEmpty(key)){
			try{
				value = Float.parseFloat(key);
			} catch(Exception e) {
			}
		}
		
		return value;
	}
	
	/**
	 * 解析字符串为Double，解析失败则返回默认值
	 * @param key
	 * @param defaultValue
	 * @return
	 */
	public static double parseDouble(String key, double defaultValue) {
		double value = defaultValue;
		if(!StringUtil.isNullOrEmpty(key)){
			try{
				value = Double.parseDouble(key);
			} catch(Exception e) {
			}
		}
		
		return value;
	}
}
