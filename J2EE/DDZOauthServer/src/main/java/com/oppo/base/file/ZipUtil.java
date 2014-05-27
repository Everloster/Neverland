/**
 * ZipUtil.java
 * com.oppo.base.file
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-17 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.file;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * ClassName:ZipUtil
 * Function: 文件解压/压缩操作
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-17  下午04:56:27
 */
public class ZipUtil {
	
	/**
	 * 解压指定的文件到指定路径
	 * @param sourcePath zip文件的路径
	 * @param destDir 解压后文件存放的路径
	 * @return
	 */
	public static boolean unzip(String sourcePath, String destDir) throws IOException {
		//判断文件是否存在且是否是普通文件
		File sFile = new File(sourcePath);
		if(!sFile.exists() || !sFile.isFile()) {
			return false;
		}
		
		//创建文件目录
		if(!FileOperate.createDir(destDir, false)) {
			return false;
		}
		
		ZipInputStream zis = null;

		try {
			zis = new ZipInputStream(new FileInputStream(sFile));
			BufferedInputStream bis = new BufferedInputStream(zis);
			
			ZipEntry ze;
			while((ze = zis.getNextEntry()) != null) {
				if(!ze.isDirectory()) {
					//得到新的文件
					File destFile = new File(destDir, ze.getName());
					//保存数据到新文件中
					FileOperate.saveStreamToFile(destFile, bis);
				}

				zis.closeEntry();
			}
			
			return true;
		} finally {
			FileOperate.close(zis);
		}
	}
	
	/**
	 * 压缩指定文件到指定的路径
	 * @param sourceFiles 源文件路径
	 * @param relatePath 源文件的相对路径，如relatePath为e:\a,
	 * 		sourceFile为e:\a\b\c.txt,则sourceFile压缩后的路径为b\c.txt
	 * @param destPath 压缩后的文件路径
	 * @return 是否压缩成功
	 * @throws IOException 
	 */
	public static boolean zip(String[] sourceFiles, String relatePath, String destPath) throws IOException {
		ZipOutputStream zos = null;
		
		int relateLen = (null == relatePath) ? 0 : relatePath.length();
		
		try {
			zos = new ZipOutputStream(new FileOutputStream(destPath));
			BufferedOutputStream bos = new BufferedOutputStream(zos);
			
			for(int i = 0; i < sourceFiles.length; i++) {
				String path = sourceFiles[i];

				//zip名称修正
				String zipName = path.substring(relateLen);
				if(zipName.startsWith(File.separator)) {
					zipName = zipName.substring(1);
				}
				
				ZipEntry zipEntry = new ZipEntry(new String(zipName.getBytes(), "ascii"));
				zos.putNextEntry(zipEntry);
				
				//读取文件内容并写入压缩流中
				FileInputStream fis = null;
				try {
					fis = new FileInputStream(path);
					
					byte[] data = new byte[4096];
					int readLen;
					while((readLen = fis.read(data)) != -1) {
						bos.write(data, 0, readLen);
					}
					
					bos.flush();
				} finally {
					FileOperate.close(fis);
				}
			}
			
			zos.finish();
			
			return true;
		} finally {
			FileOperate.close(zos);
		}
	}
	
	public static void main(String[] args) {
		try {
//			long start = System.currentTimeMillis();
//			for(int i = 0; i < 100; i++)
			ZipUtil.zip(new String[] {
					"G:\\drivers_tools\\adb_fastboot tool\\adb.exe",
					"G:\\drivers_tools\\adb_fastboot tool\\fastboot.exe",
					"G:\\drivers_tools\\adb_fastboot tool\\AdbWinUsbApi.dll"
					}, "G:\\drivers_tools", "G:\\drivers_tools\\test.zip");
//			long total = System.currentTimeMillis() - start;
//			System.out.println("耗时：" + total);
			
			ZipUtil.unzip("G:\\1\\1.zip", "g:\\des1");
			System.out.println("operate complate!");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

