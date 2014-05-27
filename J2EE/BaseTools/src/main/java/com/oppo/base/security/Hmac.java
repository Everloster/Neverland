package com.oppo.base.security;

import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;

/**
 * Hmac算法
 * @author 80036381
 * 2011-1-7
 */
public class Hmac {
	
	/**
     * 对报文进行hmac-sha1签名
     * @param aValue - 字符串
     * @param aKey - 密钥
     * @return - 签名结果，hex字符串
     */
	public static String signSHA1(String aValue, String aKey) {
		return signSHA1(aValue, aKey, OConstants.DEFAULT_ENCODING);
	}
	
	/**
     * 对报文进行hmac-sha1签名
     * @param aValue - 字符串
     * @param aKey - 密钥
     * @param encoding - 字符串编码方式
     * @return - 签名结果，hex字符串
     */
	public static String signSHA1(String aValue, String aKey, String encoding) {
		if(StringUtil.isNullOrEmpty(encoding)) {
    		encoding = OConstants.DEFAULT_ENCODING;
    	}
		
		byte[] value;
		byte[] key;
		try {
			value = aValue.getBytes(encoding);
			key = aKey.getBytes(encoding);
		} catch (UnsupportedEncodingException e1) {
			value = aValue.getBytes();
			key = aKey.getBytes();
		}
		
		SecretKeySpec signingKey = new SecretKeySpec(key, "HmacSHA1");
        Mac mac = null;
        try {
            mac = Mac.getInstance("HmacSHA1");
            mac.init(signingKey);
        } catch(Exception e) {
            return null;
        }
        
        byte[] rawHmac = mac.doFinal(value);
        return StringUtil.toHex(rawHmac);
	}
	
	/**
     * 对报文进行hmac-md5签名
     * @param aValue - 字符串
     * @param aKey - 密钥
     * @return - 签名结果，hex字符串
     */
    public static String hmacSign(String aValue, String aKey) {
    	return signMD5(aValue, aKey, OConstants.DEFAULT_ENCODING);
    }
	
	/**
     * 对报文进行hmac-md5签名
     * @param aValue - 字符串
     * @param aKey - 密钥
     * @return - 签名结果，hex字符串
     */
    public static String signMD5(String aValue, String aKey) {
    	return signMD5(aValue, aKey, OConstants.DEFAULT_ENCODING);
    }
	
	/**
     * 对报文进行hmac-md5签名
     * @param aValue - 字符串
     * @param aKey - 密钥
     * @param encoding - 字符串编码方式
     * @return - 签名结果，hex字符串
     */
    public static String signMD5(String aValue, String aKey, String encoding) {	
    	if(StringUtil.isNullOrEmpty(encoding)) {
    		encoding = OConstants.DEFAULT_ENCODING;
    	}
    	
        byte k_ipad[] = new byte[64];
        byte k_opad[] = new byte[64];
        byte keyb[];
        byte value[];
        try
        {
            keyb = aKey.getBytes(encoding);
            value = aValue.getBytes(encoding);
        }
        catch(UnsupportedEncodingException e)
        {
            keyb = aKey.getBytes();
            value = aValue.getBytes();
        }
        
        Arrays.fill(k_ipad, keyb.length, 64, (byte)54);
        Arrays.fill(k_opad, keyb.length, 64, (byte)92);
        for(int i = 0; i < keyb.length; i++)
        {
            k_ipad[i] = (byte)(keyb[i] ^ 0x36);
            k_opad[i] = (byte)(keyb[i] ^ 0x5c);
        }

        MessageDigest md = null;
        try
        {
            md = MessageDigest.getInstance("MD5");
        }
        catch(NoSuchAlgorithmException e)
        {
            return null;
        }
        md.update(k_ipad);
        md.update(value);
        byte dg[] = md.digest();        
        md.reset();
        md.update(k_opad);
        md.update(dg, 0, 16);
        dg = md.digest();
        return StringUtil.toHex(dg);
    }
    
    public static void main(String[] args) {
    	System.out.println(Hmac.signMD5("偶的神as98asdjkay7wy7yhhadasdzxcasdasd", "encryptkey"));
    	System.out.println(Hmac.signMD5("52171273860839019998509l1IaqLxH1olRcLf8m3GYcQ==", "oppo.com_oppo.com"));
    	System.out.println(Hmac.signSHA1("偶的神as98asdjkay7wy7yhhadasdzxcasdasd", "encryptkey"));
    }
}
