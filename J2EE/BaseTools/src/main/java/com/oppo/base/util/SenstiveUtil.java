/**
 * SenstiveUtil.java
 * com.oppo.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-11-16 		80053851
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 */

package com.oppo.base.util;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.oppo.base.common.FormatCheck;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:SenstiveUtil
 * 
 * Function: 敏感词过滤工具类
 * 
 * @author 80053851
 * @version
 * @since Ver 1.1
 * @Date 2011-11-16 上午11:16:55
 */
public class SenstiveUtil {

	/**
	 * 获取字符串的敏感词列表
	 * 
	 * @param pattern
	 *            敏感词正则表达式
	 * @param matcher
	 *            字符串对象
	 * @return
	 */
	public static List<String> getSenstivePhraseList(String pattern, String matcher) {
		if (!StringUtil.isNullOrEmpty(pattern, false)
				&& !StringUtil.isNullOrEmpty(matcher, false)) {
			Pattern nodePattern = FormatCheck.getPattern(pattern);
			Matcher nodeMatcher = nodePattern.matcher(matcher);
			
			List<String> list = new ArrayList<String>();
			//遍历所有的敏感词
			while (nodeMatcher.find()) {
				String sep = nodeMatcher.group();
				
				//如果不存在则添加
				if(!list.contains(sep)) {
					list.add(sep);
				}
			}

			return list;
		}
		
		return null;
	}

	/**
	 * 替换所有敏感词
	 * 
	 * @param pattern
	 *            敏感词正则表达式
	 * @param matcher
	 *            字符串对象
	 * @param replacement
	 *            替换符号
	 * @return
	 */
	public static String replaceAllSenstivePhrase(String pattern,
			String matcher, String replacement) {
		if (!StringUtil.isNullOrEmpty(pattern)
				&& !StringUtil.isNullOrEmpty(matcher)) {
			Pattern nodePattern = FormatCheck.getPattern(pattern);
			Matcher nodeMatcher = nodePattern.matcher(matcher);
			
			return nodeMatcher.replaceAll(replacement);
		}
		return matcher;
	}

	public static void main(String[] args) {
		// pattern不能以"|"符号开始或结束
		String pattern = "王八蛋|淫荡|妈的";

		String matcher = "妈的xxx妈的x妈的xx淫荡xxx";
		
		System.out.println(getSenstivePhraseList(pattern, matcher));
		System.out.println(replaceAllSenstivePhrase(pattern, matcher, "*"));
	}
}
