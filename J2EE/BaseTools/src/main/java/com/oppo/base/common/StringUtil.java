package com.oppo.base.common;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.UUID;

/**
 * 和String相关的工具类
 * @author 80036381
 * 2011-1-7
 */
public class StringUtil {

	private static final char[] RANDOM_CHARS = "abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ@._()".toCharArray();

	/**
	 * 空的String数组
	 */
	public static final String[] EMPTY_ARRAY = new String[0];

	/**
	 * 生成指定长度的随机字符串
	 * @param length
	 * @return
	 */
	public static String generateRandomString(int length) {
		return generateRandomString(RANDOM_CHARS, length);
	}

	/**
	 * 生成指定长度指定范围的随机字符串
	 * @param randChars
	 * @param length
	 * @return
	 */
	public static String generateRandomString(char[] randChars, int length) {
		if(length < 1) {
			return null;
		}

		Random rand = new Random();
		int rLen = randChars.length;
		char[] newStr = new char[length];
		for(int i = 0; i < length; i++) {
			newStr[i] = randChars[rand.nextInt(rLen)];
		}

		return new String(newStr);
	}

	/**
	 * 生成全局唯一值
	 * @return
	 */
	public static String generateGUID() {
		return replaceWithBlank(UUID.randomUUID().toString(), '-');
	}

	/**
	 * 将字符串中的特定字符替换为空（即去掉此字符）
	 * @param
	 * @return
	 */
	public static String replaceWithBlank(String value, char oldChar) {
		if(null == value || value.length() == 0) {
			return value;
		}

		int len = value.length();
		int index = 0;
		char[] newValue = new char[len];
		for(int i = 0; i < len; i++) {
			char currentChar = value.charAt(i);
			if(currentChar != oldChar) {
				newValue[index++] = currentChar;
			}
		}

		return new String(newValue, 0, index);
	}

	/**
	 * 将字节数组转换为16进制形式
	 * @param input
	 * @return
	 */
	public static String toHex(byte[] input) {
		if(input == null)
            return null;
        StringBuilder output = new StringBuilder(input.length * 2);
        for(int i = 0; i < input.length; i++){
            int current = input[i] & 0xff;
            //小于16的需要补充一位(共2位)
            if(current < 16) {
                output.append('0');
            }

            output.append(Integer.toString(current, 16));
        }

        return output.toString();
	}

	/**
	 * 填充指定字符到字符串左边到指定长度
	 * @param orignalValue
	 * @param padChar	需要填充的字符
	 * @param length	填充后的长度
	 * @return
	 */
	public static String padLeft(String orignalValue, char padChar, int length) {
		return pad(orignalValue, padChar, length, true);
	}

	/**
	 * 填充指定字符到字符串右边到指定长度
	 * @param orignalValue
	 * @param padChar	需要填充的字符
	 * @param length	填充后的长度
	 * @return
	 */
	public static String padRight(String orignalValue, char padChar, int length) {
		return pad(orignalValue, padChar, length, false);
	}

	/**
	 * 填充指定字符到字符串的指定位置知道达到指定长度
	 * @param orignalValue
	 * @param padChar	需要填充的字符
	 * @param length	填充后的长度
	 * @param isLeft	是否填充在左边
	 * @return
	 */
	public static String pad(String orignalValue, char padChar, int length, boolean isLeft) {
		//null作为空字符串
		if(null == orignalValue) {
			orignalValue = OConstants.EMPTY_STRING;
		}

		//判断长度是否比需要的长度短，只有在短的情况下才会进行处理
		int oldLen = orignalValue.length();
		if(oldLen >= length) {
			return orignalValue;
		} else {
			int padLen = length - oldLen;
			//得到需要添加的字符
			char[] appendChars = new char[padLen];
			Arrays.fill(appendChars, padChar);

			char[] newChars = new char[length];
			char[] orignalChars = orignalValue.toCharArray();
			if(isLeft) {
				System.arraycopy(appendChars, 0, newChars, 0, padLen);
				System.arraycopy(orignalChars, 0, newChars, padLen, oldLen);
			} else {
				System.arraycopy(orignalChars, 0, newChars, 0, oldLen);
				System.arraycopy(appendChars, 0, newChars, oldLen, padLen);
			}

			return new String(newChars);
		}
	}

	/**
	 * 验证字符串是否为空或null
	 * @param value
	 * @return
	 */
	public static boolean isNullOrEmpty(String value) {
		return isNullOrEmpty(value, true);
	}

	/**
	 * 验证字符串是否为空或null
	 * @param value
	 * @return
	 */
	public static boolean isNullOrEmpty(String value, boolean trim) {
		int len;
		if(null == value || (len = value.length()) == 0) {
			return true;
		}

		if(!trim) {
			return false;
		}

		for(int i = 0; i < len; i++) {
			if(value.charAt(i) != ' ') {
				return false;
			}
		}

		return true;
	}

	public static String subString(String text, int start, int length) {
		if(null == text || text.length() == 0) {
			return OConstants.EMPTY_STRING;
		}

		int endPos = text.length();
		//开始位置大于总长度，则返回空
		if(start >= endPos) {
			return OConstants.EMPTY_STRING;
		}

		//
		if(start + length > endPos) {
			length = endPos - start;
		}

		return text.substring(start, start + length);
	}


	/**
	 * 按照指定长度进行截取
	 * @param
	 * @return
	 */
	public static String getStringInLength(String value, int length) {
		if(StringUtil.isNullOrEmpty(value) || value.length() <= length) {
			return value;
		} else {
			return value.substring(0, length);
		}
	}

	public static long getIPNumber(String clientIP) {
		if(StringUtil.isNullOrEmpty(clientIP)) {
			return 0;
		} else {
			long ipNumber = 0;
			int indexPre = 0;
			int index;
			while((index = clientIP.indexOf('.', indexPre)) > 0) {
				try {
					ipNumber = 256 * ipNumber + Integer.parseInt(clientIP.substring(indexPre, index));
				} catch(Exception e) {
					return 0;
				}

				indexPre = index + 1;
			}

			try {
				ipNumber = 256 * ipNumber + Integer.parseInt(clientIP.substring(indexPre));
			} catch(Exception e) {
				return 0;
			}

			return ipNumber;
		}
	}

	/**
	 * 将字符串中的指定旧值替换成指定新值
	 * @param input
	 * @param oldValue 被替换的旧值
	 * @param newValue 替换成的新值
	 * @return
	 */
	public static String replaceLast(String input, String oldValue, String newValue) {
		int lIndex = input.lastIndexOf(oldValue);
		if(lIndex < 0) {
			return input;
		}

		StringBuilder sb = new StringBuilder(
				input.length() - oldValue.length() + newValue.length());
        sb.append(input.substring(0, lIndex));
        sb.append(newValue);
        sb.append(input.substring(lIndex + oldValue.length()));

        return sb.toString();
	}

	/**
	 * 将字符串中的指定旧值替换成指定新值
	 *
	 * @param input 原字符串
	 * @param oldValues 被替换的旧值列表
	 * @param newValues 替换成的新值列表
	 * @return
	 */
	public static String replaceChars(String input, String[] oldValues, String[] newValues) {

		// 参数为空则返回原字符串
		if(StringUtil.isNullOrEmpty(input) || oldValues == null || newValues == null) {
			return input;
		}

		int size = oldValues.length;
		if(newValues.length != size) {
			return input;
		}

		for(int i = 0; i < size; i++) {
			input = input.replaceAll(oldValues[i], newValues[i]);
		}

		return input;
	}

	/**
	 * 将字符串中的指定旧值替换成指定新值
	 *
	 * @param input 原字符串
	 * @param oldValue 被替换的旧值
	 * @param newValue 替换成的新值
	 * @return
	 */
	public static String replaceChar(String input, String oldValue, String newValue) {

		// 参数为空则返回原字符串
		if(StringUtil.isNullOrEmpty(input) || StringUtil.isNullOrEmpty(oldValue) || StringUtil.isNullOrEmpty(newValue)) {
			return input;
		}

		return input.replaceAll(oldValue, newValue);
	}

	/**
	 * 直接针对String的split，不支持正则表达式
	 * @param
	 * @return
	 */
	public static String[] split(String value, char splitChar) {
		if(null == value) {
			return null;
		}

		int len = value.length();
		if(len == 0) {
			return EMPTY_ARRAY;
		}

		int lastFromIndex = 0; //最近一次查询的其实位置
		int index;
		ArrayList<String> result = new ArrayList<String>();
		while((index = value.indexOf(splitChar, lastFromIndex)) != -1) {
			if(lastFromIndex != index) {
				result.add(value.substring(lastFromIndex, index));
			}

			lastFromIndex = index + 1;
		}

		if(lastFromIndex >= 0 && lastFromIndex != len) {
			result.add(value.substring(lastFromIndex));
		}

		return result.toArray(new String[result.size()]);
	}

	/**
	 * 直接针对String的split，不支持正则表达式
	 * @param
	 * @return
	 */
	public static String[] split(String value, String splitString) {
		if(null == value) {
			return null;
		}

		int len = value.length();
		if(len == 0) {
			return EMPTY_ARRAY;
		}

		int splitStringLen = splitString.length();
		if(splitStringLen == 0) {
			return new String[] { value };
		}

		int lastFromIndex = 0; //最近一次查询的其实位置
		int index;
		ArrayList<String> result = new ArrayList<String>();
		while((index = value.indexOf(splitString, lastFromIndex)) != -1) {
			if(lastFromIndex != index) {
				result.add(value.substring(lastFromIndex, index));
			}

			lastFromIndex = index + splitStringLen;
		}

		if(lastFromIndex >= 0 && lastFromIndex != len) {
			result.add(value.substring(lastFromIndex));
		}

		return result.toArray(new String[result.size()]);
	}

	public static void main(String[] args) {
//		String s = "172.16.1.1:8080;172.16.1.2:8080";
//		p(StringUtil.split(s, ";"));
//		p(StringUtil.split(s, ';'));
//		p(s.split(";"));

		System.out.println(StringUtil.generateRandomString(16));
		System.out.println(StringUtil.generateGUID());

		///System.out.println(StringUtil.replaceLast("woshiashiyige", "shi", "bushi"));
//		String[] oldValues = {"\"", "<", "/>"};
//		String[] newValues = {"&#34;", "&#60;", "/&#62;"};
//		System.out.println(StringUtil.replaceChars("dffd\"sf\"df<d<fsdfs/>df", oldValues, newValues));
	}

//	private static void p(String[] a) {
//		for(String s : a) {
//			System.out.println(s);
//		}
//		System.out.println("-----------");
//	}
}
