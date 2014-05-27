package com.nearme.apkchannel.security.test;

import it.sauronsoftware.base64.Base64;

import java.io.UnsupportedEncodingException;

import junit.framework.Assert;

import org.junit.Test;

import com.nearme.apkchannel.security.AESTool;

public class AESToolTest {

    @Test
    public void testNormal() throws UnsupportedEncodingException {
        
        String content = "asdfasdfas";
        // 加密
        byte[] encryptResult = AESTool.encrypt(content);
        
        System.out.println(encryptResult.length);
        
        printBytes(encryptResult);
        
        System.out.println(new String(encryptResult).getBytes().length);
        
        printBytes(new String(encryptResult).getBytes());
        
        String encryptResultStr = new String(Base64.encode(encryptResult));
        
        printBytes(Base64.decode(encryptResultStr.getBytes()));
        
        // 解密
        byte[] decryptResult = AESTool.decrypt(Base64.decode(encryptResultStr.getBytes()));
        
        Assert.assertEquals(content, new String(decryptResult));
        
        System.out.println(content.equals(new String(decryptResult)));
    }
    
    public static void main(String[] args) {
        try {
            new AESToolTest().testNormal();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }
    
    private static void printBytes(byte[] bytes) {
        for (int i = 0; i < bytes.length; i++) {
            System.out.print(bytes[i]);
            System.out.print(" ");
        }
        System.out.println();
    }
}
