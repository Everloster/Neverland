package com.nearme.apkchannel.test;

import it.sauronsoftware.base64.Base64;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import com.nearme.apkchannel.ChannelTool;
import com.nearme.apkchannel.security.AESTool;

public class ChannelToolTest {
    
    private static final String NEVER_LAND_JNI_APK = "NeverLandJni.apk";
    
    final String prefix = "GL-";
    final String code = "sdfssdfsadssdfsdfa";
    final String encodeString = ChannelTool.encodeMethod(prefix, code);
    
    @Before
    public void setUp() throws UnsupportedEncodingException {
        ChannelTool.main(new String[]{prefix, code, NEVER_LAND_JNI_APK});
    }

    @Test
    public void testChannelFileCreateInLocalAndInApk() {
        byte[] encodedData = null;
        byte[] tmp = new byte[128];
        try {
            InputStream in = new FileInputStream("update" + File.separator + "assets" + File.separator + "channel");
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            int count = 0;
            while ((count = in.read(tmp)) != -1) {
                out.write(tmp, 0, count);
            }
            encodedData = out.toByteArray();
            out.flush();
            in.close();
            out.close();
            
            Assert.assertEquals(encodeString, new String(encodedData));
            String encodedHexStr = new String(Base64.decode(encodedData)).substring(3);
            Assert.assertEquals(code, new String(AESTool.decrypt(Base64.decode(encodedHexStr.getBytes()))));
            
            Assert.assertEquals(prefix + code, ChannelTool.readChannelFromApk(NEVER_LAND_JNI_APK + ".channel.apk"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
