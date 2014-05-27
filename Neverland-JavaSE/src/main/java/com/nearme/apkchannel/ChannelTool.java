package com.nearme.apkchannel;

import it.sauronsoftware.base64.Base64;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;

import com.nearme.apkchannel.security.AESTool;

public class ChannelTool {

    private static String CHANNEL_DIR = "update" + File.separator + "assets";
    private static String CHANNEL_FILE = "update" + File.separator + "assets" + File.separator + "channel";

    static {
        createDir(CHANNEL_DIR);
    }

    private static void createDir(String dir) {
        File f = new File(dir);
        if (!f.exists()) {
            f.mkdirs();
        }
    }

    private static void writeChannelFile(String fileName, byte[] content, boolean needOverride) {
        try {
            if (needOverride) {
                new File(fileName).delete();
            }
            File randomFile = new File(fileName);
            FileOutputStream out = new FileOutputStream(randomFile);
            out.write(content, 0, content.length);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void exeNativeCmd(String cmd) {
        Runtime rt = Runtime.getRuntime();
        Process p = null;
        try {
            p = rt.exec(cmd);
            BufferedReader input = new BufferedReader(new InputStreamReader(p.getInputStream()));
            BufferedReader errorInput = new BufferedReader(new InputStreamReader(p.getErrorStream()));
            String line = "";
            while ((line = input.readLine()) != null) {
                System.out.println(line);
            }
            while ((line = errorInput.readLine()) != null) {
                System.out.println(line);
            }
            input.close();
            errorInput.close();
            p.waitFor();
            int i = p.exitValue();
            if (i == 0) {
                System.out.println("update and sign apk success.");
            } else {
                System.out.println("update and sign apk failure.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws UnsupportedEncodingException {
        if (args.length != 3) {
            System.out.println("Invalid params!");
            System.out.println("Please input : channel.jar 渠道号前缀  渠道号（加密）apkFilePath");
            return;
        }
        final String prefix = args[0];
        final String code = args[1];
        final String apkPath = args[2];
        final String encodeString = encodeMethod(prefix, code);
        // write channel file to local.
        System.out.println("Begin to write channel file to local : " + encodeString);
        writeChannelFile(CHANNEL_FILE, encodeString.getBytes(), true);
        // exe native cmd to update and sign apk.
        try {
            exeNativeCmd(getApkShellCmd(apkPath, readZipFile(apkPath)));
        } catch (Exception e) {
            e.printStackTrace();
        }
        // check apk's channel 
        try {
            String get_channel = readChannelFromApk(apkPath + ".channel.apk");
            System.out.println("Check apk's channel : " + get_channel);
            System.out.println("Check result : " + (prefix + code).equals(get_channel));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 加密方法步骤
     * 
     * 1.对string code进行AES加密产生数组a
     * 2.对数组a进行base64加密产生数组b
     * 3.用数组b产生string c
     * 4.用string prefix和string c进行base64加密后最终得到密文string
     * 
     * @param prefix
     * @param code
     * @return 密文
     * @throws UnsupportedEncodingException 
     */
    public static String encodeMethod(final String prefix, final String code) {
        final byte[] encodeBytes = AESTool.encrypt(code);
        String transportStr = new String(Base64.encode(encodeBytes));
        return Base64.encode(prefix + transportStr);
    }

    private static String getApkShellCmd(String apkPath, String[] params) {
        StringBuilder sb = new StringBuilder("sh apkchannel");
        sb.append(" ").append(apkPath);
        for (String tag : params) {
            sb.append(" ");
            sb.append(tag);
        }
        System.out.println(sb.toString());
        return sb.toString();
    }

    @SuppressWarnings("resource")
    private static String[] readZipFile(String file) throws Exception {
        InputStream in = new BufferedInputStream(new FileInputStream(file));
        ZipInputStream zin = new ZipInputStream(in);
        ZipEntry ze;
        String[] fileList = { "", "", "" };
        while ((ze = zin.getNextEntry()) != null) {
            if (!ze.isDirectory() && ze.getName().contains("META-INF")) {
                if (ze.getName().endsWith("RSA")) {
                    fileList[0] = ze.getName();
                }
                if (ze.getName().endsWith("SF")) {
                    fileList[1] = ze.getName();
                }
                if (ze.getName().endsWith("MF")) {
                    fileList[2] = ze.getName();
                }
            }
        }
        zin.closeEntry();
        return fileList;
    }

    public static String readChannelFromApk(String file) throws IOException {
        String result = null;
        ZipFile zf = new ZipFile(file);
        InputStream in = new BufferedInputStream(new FileInputStream(file));
        ZipInputStream zin = new ZipInputStream(in);
        ZipEntry ze;
        while ((ze = zin.getNextEntry()) != null) {
            if (ze.isDirectory()) {
                continue;
            }
            if (!ze.isDirectory() && ze.getName().toLowerCase().equals("assets/channel")) {
                long size = ze.getSize();
                if (size > 0) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zf.getInputStream(ze)));
                    String line;
                    while ((line = br.readLine()) != null) {
                        result = line;
                    }
                    br.close();
                    break;
                }
            }
        }
        zin.closeEntry();
        in.close();
        zf.close();
        System.out.println("Get channel from apk : " + result);
        return decodeChannelString(result, 3);
    }

    public static String decodeChannelString(String data, int preLength) {
        if (data == null) {
            return null;
        }
        String dataStr = Base64.decode(data);
        final String prefix = dataStr.substring(0, preLength);
        final String code = dataStr.substring(3);
        return prefix + new String(AESTool.decrypt(Base64.decode(code.getBytes())));
    }
}
