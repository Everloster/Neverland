/**
 * SHA1.java
 * com.oppo.base.security
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-7-17 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.oppo.base.security;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.security.NoSuchAlgorithmException;

import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;
import com.oppo.base.file.FileOperate;

/**
 * ClassName:SHA1 <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-7-17  下午03:56:37
 */
public class SHA1 {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * md5加密
	 * @param text 加密字符串
	 * @param encoding 字符编码,传空则使用默认编码
	 * @return
	 */
	public static String sha1(String text) {
		return sha1(text, OConstants.DEFAULT_ENCODING);
	}
	
	/**
	 * md5加密
	 * @param text 加密字符串
	 * @param encoding 字符编码,传空则使用默认编码
	 * @return
	 */
	public static String sha1(String text, String encoding) {
		if(text == null) {
			return null;
		}
		
		if(StringUtil.isNullOrEmpty(encoding)) {
    		encoding = OConstants.DEFAULT_ENCODING;
    	}
		
		try {
			return sha1(text.getBytes(encoding));
		} catch (UnsupportedEncodingException e) {
			return null;
		}
	}
	
	/**
	 * 对字节数组的sha1加密
	 * @param 
	 * @return
	 */
	public static String sha1(byte[] data) {
		return StringUtil.toHex(sha1ToBytes(data));
	}
	
	/**
	 * 对字节数组的sha1加密
	 * @param 
	 * @return
	 */
	public static byte[] sha1ToBytes(byte[] data) {
		try {
			java.security.MessageDigest md = java.security.MessageDigest.getInstance("sha-1");
			md.update(data);
			return md.digest();
		} catch (NoSuchAlgorithmException e) {
			return null;
		}
	}
	
	/**
	 * 对InputStream的sha1加密
	 * @param 
	 * @return
	 */
	public static String getInputStreamSHA1(InputStream inputStream) throws Exception {
		java.security.MessageDigest md = java.security.MessageDigest.getInstance("sha-1");
		
		//读取InputStream中的字节
		int cacheLen = 4096;
		byte[] cData = new byte[cacheLen];
		int readLen;
		while((readLen = inputStream.read(cData, 0, cacheLen)) != -1) {
			md.update(cData, 0, readLen);
		}
		
		//进行md5计算
		return StringUtil.toHex(md.digest());
	}
	
	/**
	 * 对文件的sha1加密
	 * @param 
	 * @return
	 */
	public static String getFileSHA1(File file) {
		if(null == file || !file.exists()) {
			return null;
		}
		
		FileInputStream fis = null;
		try {
			fis = new FileInputStream(file);
			return getInputStreamSHA1(fis);
		} catch(Exception ex){
			return null;
		} finally {
			FileOperate.close(fis);
		}
	}
	
	public static void main(String[] args) {
		System.out.println(SHA1.sha1("我的"));
	}
}

