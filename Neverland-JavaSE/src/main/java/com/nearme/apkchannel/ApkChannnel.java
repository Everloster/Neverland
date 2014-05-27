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
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class ApkChannnel {
    
    
    static String channelDir = "update\\assets";
    static String channelFile = "update\\assets\\channel";

    
    static {
        createDir(channelDir);
    }

    public static void createDir(String dir) {
        File f = new File(dir);
        if (!f.exists()) {
            f.mkdirs();
        }
    }

    public static void writeFile(String fileName, byte[] content) {
        try {
            File randomFile = new File(fileName);
            FileOutputStream out = new FileOutputStream(randomFile);
            out.write(content, 0, content.length);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void genApkChannel(String apk, String channel, String channelValue, String[] names) throws Exception {
        byte[] encodedData = Base64.encode(channel.getBytes());
        writeFile(channelFile, encodedData);
        Runtime rt = Runtime.getRuntime();
        Process p = null;
        try {
            File file = new File(channelValue + ".apk");
            if (file.exists()) {
                file.delete();
            }
            final StringBuilder sb = new StringBuilder("cmd.exe /C start ApkChannel.bat " + apk + " " + channelValue + ".apk");
            for (String s : names) {
                sb.append(" " + s);
            }
            System.out.println(sb.toString());
            p = rt.exec(sb.toString());
            p.waitFor();
            int i = p.exitValue();
            if (i == 0) {
                System.out.println("执行完成.");
            } else {
                System.out.println("执行失败.");
            }
            Thread.sleep(10000);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("参数为空.");
            System.out.println("使用方式  channel.jar 渠道号前缀（不加密） 渠道号后缀（加密） *.apk");
            return;
        }
        try {
            String iniFile = "cfg.ini";
            String apkFile = args[0];
            parseIni(apkFile, iniFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void parseIni(String apkFile, String iniFile) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(iniFile)));
        String line;
        while ((line = reader.readLine()) != null) {
            parseLine(apkFile, line);
        }
        reader.close();
    }

    private static void parseLine(String apkFile, String line) {
        line = line.trim();
        if (line.matches(".*=.*")) {
            int i = line.indexOf('=');
            String name = line.substring(0, i);
            String value = line.substring(i + 1);
            System.out.println("" + name + "::" + value);
            try {
                String[] fileString = readZipFile(apkFile);
                ApkChannnel.genApkChannel(apkFile, name, value, fileString);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
    
    @SuppressWarnings("resource")
    public static String[] readZipFile(String file) throws Exception {
        InputStream in = new BufferedInputStream(new FileInputStream(file));  
        ZipInputStream zin = new ZipInputStream(in);  
        ZipEntry ze;  
        String[] fileList = {"","",""};
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
}
