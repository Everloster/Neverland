package com.oppo.base.security;

import java.io.File;

import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;

import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;
import com.oppo.base.file.FileOperate;

/**
 * 提供3DES加解密功能
 * @author 80036381
 * 2011-1-7
 */
public class Des {
	//DES算法名称  
	private static final String DES_KEY_ALGORITHM = "DES";  
	//DES算法名称/加密模式/填充方式  
	private static final String DES_CIPHER_ALGORITHM = "DES/ECB/PKCS5Padding"; 
	
	/**
	 * des加密
	 * 
	 * @param value
	 * @param key
	 * @return
	 */
	public static byte[] encrypt(String value, String key, String encoding) {
		if (StringUtil.isNullOrEmpty(encoding)) {
			encoding = OConstants.DEFAULT_ENCODING;
		}
		
		try {
			//key = key.substring(0, 24);
			return encrypt(value.getBytes(encoding), key.getBytes(encoding));
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
	
	/**
	 * des加密
	 * @param 
	 * @return
	 */
	public static byte[] encrypt(byte[] value, byte[] key) throws Exception {
		SecretKey deskey = new SecretKeySpec(key, DES_KEY_ALGORITHM);
		Cipher c1 = Cipher.getInstance(DES_CIPHER_ALGORITHM);
		c1.init(Cipher.ENCRYPT_MODE, deskey);
		return c1.doFinal(value);
	}

	/**
	 * des解密
	 * 
	 * @param value
	 * @param key
	 * @return
	 */
	public static byte[] decrypt(String value, String key, String encoding) {
		if (StringUtil.isNullOrEmpty(encoding)) {
			encoding = OConstants.DEFAULT_ENCODING;
		}

		try {
			return decrypt(value.getBytes(encoding), key.getBytes(encoding));
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
	
	public static byte[] decrypt(byte[] value, byte[] key) throws Exception {
		SecretKey deskey = new SecretKeySpec(key, DES_KEY_ALGORITHM);
		Cipher c1 = Cipher.getInstance(DES_CIPHER_ALGORITHM);
		c1.init(Cipher.DECRYPT_MODE, deskey);
		
		return c1.doFinal(value);
	}
	
	public static void main(String[] args) throws Exception {
		long s = System.currentTimeMillis();
		byte[] value = FileOperate.getShareBytes(new File("G:\\OPPONewSSO2.war"));
		byte[] a = Des.encrypt(value, "23232334".getBytes());
		byte[] b = Des.decrypt(a, "23232334".getBytes());
		FileOperate.saveContentToFile("G:\\2.txt", a);
		FileOperate.saveContentToFile("G:\\OPPONewSSO2_a.war", b);
		
		long t = System.currentTimeMillis() - s;
	    System.out.println(t);
	}
}
