/**
 * Copyright (c) 2011 NearMe, All Rights Reserved.
 * FileName:KeyGenerator.java
 * ProjectName:OBaseTool
 * PackageName:com.oppo.base.common
 * Create Date:2011-9-5
 * History:
 *   ver	date	  author		desc	
 * ────────────────────────────────────────────────────────
 *   1.0	2011-9-5	  80054252		
 *
 * 
*/

package com.oppo.base.common;

import java.math.BigInteger;
import java.util.Random;
import java.util.UUID;


/**
 * ClassName:KeyGenerator
 * @author   80054252
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-9-5  上午10:28:41
 */
public class KeyGenerator {

	/**
	 * 生成一个全局唯一的KEY
	 * 
	 * @return 一个全局唯一的KEY
	 */
	public static String generateKey() {
		
//		// 获得一个长度为32的不重复的字符串
//		String key = StringUtil.generateGUID();
		// 缩短key的长度
		String key = new BigInteger(StringUtil.generateGUID(), 16).toString(Character.MAX_RADIX);
		
		// 将其中的一些字符转换为大写
		char[] array = randomToUpperCase(key, 6, 15, key.length());
		// 返回转换后的结果
		return String.valueOf(array);
	}
	
	/**
	 * 生成一个全局唯一的SECRET
	 * 
	 * @return 一个全局唯一的SECRET
	 */
	public static String generateSecret() {
		
		// 获得一个长度为32的不重复的字符串
		String secret = StringUtil.generateGUID();
		// 将其中的一些字符转换为大写
		char[] array = randomToUpperCase(secret, 11, 20, secret.length());
		// 返回转换后的结果
		return String.valueOf(array);
	}

	/**
	 * Function Description here
	 * @param min 转为大写的最小字符数
	 * @param max 转为大写的最大字符数
	 * @param range 随机数范围
	 * @return
	 */
	
	private static char[] randomToUpperCase(String secret, int min, int max, int range) {
		
		// 生成字符数字
		char[] array = secret.toCharArray();
		
		// 产生一个随机数生成器
		Random random = new Random(UUID.randomUUID().hashCode());
		// 取得一个min-max随机数，设为该字符串中大写字母的个数
		int len = min + random.nextInt(max - min + 1);
		// 随机将其中至少min个字符转换成大写（若指定位置的字符为数字则不转换）
		for (int i = 0; i < len; i++)
        {
        	// 生成一个0-31的随机数
            int index = random.nextInt(range);
            // 取得指定位置的字符
            String ch = String.valueOf(array[index]);
            // 判断指定位置的字符是否是小写字母，若是小写字母，则将其转为大写
            if(FormatCheck.checkLowerCaseText(ch)) {
            	array[index] -= 32;
            }
        }
		return array;
	}
	
	public static void main(String[] args) {
		System.out.println(KeyGenerator.generateKey());
	}
}

