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
 * MD5加密
 * @author 80036381
 * 2011-1-7
 */
public class MD5 {
	/**
	 * md5加密
	 * @param text 加密字符串
	 * @param encoding 字符编码,传空则使用默认编码
	 * @return
	 */
	public static String md5(String text) {
		return md5(text, OConstants.DEFAULT_ENCODING);
	}
	
	/**
	 * md5加密
	 * @param text 加密字符串
	 * @param encoding 字符编码,传空则使用默认编码
	 * @return
	 */
	public static String md5(String text, String encoding) {
		if(text == null) {
			return null;
		}
		
		if(StringUtil.isNullOrEmpty(encoding)) {
    		encoding = OConstants.DEFAULT_ENCODING;
    	}
		
		try {
			return md5(text.getBytes(encoding));
		} catch (UnsupportedEncodingException e) {
			return null;
		}
	}
	
	/**
	 * 对字节数组的MD5加密
	 * @param 
	 * @return
	 */
	public static String md5(byte[] data) {
		return StringUtil.toHex(md5ToBytes(data));
	}
	
	/**
	 * 对字节数组的MD5加密
	 * @param 
	 * @return
	 */
	public static byte[] md5ToBytes(byte[] data) {
		try {
			java.security.MessageDigest md = java.security.MessageDigest.getInstance("MD5");
			md.update(data);
			return md.digest();
		} catch (NoSuchAlgorithmException e) {
			return null;
		}
	}
	
	/**
	 * 对InputStream的MD5加密
	 * @param 
	 * @return
	 */
	public static String getInputStreamMD5(InputStream inputStream) throws Exception {
		java.security.MessageDigest md = java.security.MessageDigest.getInstance("MD5");
		
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
	 * 对文件的MD5加密
	 * @param 
	 * @return
	 */
	public static String getFileMD5(File file) {
		if(null == file || !file.exists()) {
			return null;
		}
		
		FileInputStream fis = null;
		try {
			fis = new FileInputStream(file);
			return getInputStreamMD5(fis);
		} catch(Exception ex){
			return null;
		} finally {
			FileOperate.close(fis);
		}
	}
	
	public static void main(String[] args) {
		long start = System.currentTimeMillis();
		for(int i = 0; i < 10; i++) {
			System.out.println(MD5.getFileMD5(new File("G:\\gamecenter.log")));
		}
		long end = System.currentTimeMillis();
		System.out.println((end - start));
	}
}
