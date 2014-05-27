package com.oppo.base.common;

import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/**
 * 提供基本的格式检查
 * @author 80036381
 * 2011-1-7
 */
public class FormatCheck {
	
	private static final ConcurrentHashMap<String, Pattern> PATTERN_MAP = new ConcurrentHashMap<String, Pattern>();
	
	private static final Pattern PATTERN_EMAIL = Pattern.compile(
			"^(\\w+((-\\w+)|(\\.\\w+))*)+\\w+((-\\w+)|(\\.\\w+))*@[A-Za-z0-9]+((\\.|-)[A-Za-z0-9]+)*\\.[A-Za-z0-9]+$");
	
	private static final Pattern PATTERN_MOBILE = Pattern.compile("^1[0-9]{10}$");
	
	private static final Pattern PATTERN_NUMERIC = Pattern.compile("^[0-9]+$");
	
	private static final Pattern PATTERN_LOWERCASE = Pattern.compile("^[a-z]+$");
	
	/**
	 * 验证邮箱是否合法
	 * @param email
	 * @return
	 */
	public static boolean checkEmail(String email) {
		return checkTextWithReg(email, PATTERN_EMAIL);
	}
	
	/**
	 * 验证手机号码是否合法
	 * @param 
	 * @return
	 */
	public static boolean checkMobile(String mobile) {
		return checkTextWithReg(mobile, PATTERN_MOBILE);
	}
	
	/**
	 * 检测用户名是否合法
	 * @param 
	 * @return
	 */
	public static boolean checkUserName(String userName) {
		return !StringUtil.isNullOrEmpty(userName);
	}
	
	/**
	 * 验证一个字符串是否是数字字符串
	 * @param 
	 * @return
	 */
	public static boolean checkNumeric(String text) {
		return checkTextWithReg(text, PATTERN_NUMERIC);
	}
	
	/**
	 * 验证一个字符串是否只包含a-z
	 * @param 
	 * @return
	 */
	public static boolean checkLowerCaseText(String text) {
		return checkTextWithReg(text, PATTERN_LOWERCASE);
	}
	
	/**
	 * 检测指定文本是否符合指定的正则表达式
	 * Function Description here
	 * @param 
	 * @return
	 */
	public static boolean checkTextWithReg(String text, String reg) {
		if(StringUtil.isNullOrEmpty(text, false)) {
			return false;
		}
		
		Pattern regex = getPattern(reg);
		return regex.matcher(text).matches();
	}
	
	/**
	 * 检测指定文本是否符合指定的正则表达式
	 * Function Description here
	 * @param 
	 * @return
	 */
	public static boolean checkTextWithReg(String text, Pattern regex) {
		if(StringUtil.isNullOrEmpty(text, false)) {
			return false;
		}
		
		return regex.matcher(text).matches();
	}
	
	public static Pattern getPattern(String regex) {
		Pattern pattern = PATTERN_MAP.get(regex);
		if(null == pattern) {
			pattern = Pattern.compile(regex);
			PATTERN_MAP.put(regex, pattern);
		}
		
		return pattern;
	}
	
	public static void main(String[] args) {
		System.out.println(checkEmail("ybDqHIpO@126.com"));
		System.out.println(checkMobile("11111111116"));
		
		System.out.println(checkTextWithReg("<head>asd</head>","<head>.*</head>"));
	}
}
