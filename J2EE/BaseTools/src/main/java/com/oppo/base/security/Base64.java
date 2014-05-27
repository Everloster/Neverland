package com.oppo.base.security;

import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;

/**
 * 提供Base64编码解码
 * @author 80036381
 * 2011-1-7
 */
public class Base64 {
	/**
	 * base64编码
	 * @param data 要编码的字符
	 * @param encoding 字符编码，传空则使用默认编码
	 * @return 
	 */
	public static String base64Encode(String data, String encoding) {
		if(StringUtil.isNullOrEmpty(encoding)) {
    		encoding = OConstants.DEFAULT_ENCODING;
    	}

		try{
			return base64Encode(data.getBytes(encoding));
		}catch(Exception e) {
			return null;
		}
	}
	
	/**
	 * base64编码
	 * @param data 要编码的字符
	 * @return
	 */
	public static String base64Encode(byte[] data) {
		try {
			return org.apache.commons.codec.binary.Base64.encodeBase64String(data);
		} catch (Exception e) {
			return null;
		}
	}

	/**
	 * base64解码
	 * @param data 要解码的字符
	 * @return
	 */
	public static byte[] base64Decode(String data){
		try {
			return org.apache.commons.codec.binary.Base64.decodeBase64(data);
		} catch (Exception e) {
			return null;
		}
	}
	
	public static void main(String[] args) {
		byte[] data = base64Decode("c2VydmVyPUFQUFNUT1JFMTt1c2VyIGlkPWFwcHN0b3JlO3Bhc3N3b3JkPU1lSE5CUFBXbXlQNTMySmI7TWF4IFBvb2wgU2l6ZT01MTI7TWluIFBvb2wgU2l6ZT0yMA==");
		System.out.println(new String(data));
	}
}
