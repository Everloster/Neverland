/**
 * FileHelper.java
 * com.nearme.base.file
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-12-8 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.file;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import com.oppo.base.file.FileOperate;

/**
 * ClassName:FileHelper <br>
 * Function: 文件相关操作 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-12-8  下午04:22:59
 */
public class FileHelper {
	/**
	 * 读或写的缓冲字节长度
	 */
	private static final int BUFFER_LENGTH = 8192;
	
	/**
	 * 将文件内容读取到预分配好的字节数组中,提高文件读取的速度 <br/>
	 * 注意，linux下可能无法读取到文件大小，不能用此方法读取长度超过最大int值的文件
	 * 
	 * @param filePath 文件路径
	 * @return
	 */
	public static byte[] getFileBytes(String filePath) throws IOException {
		return getFileBytes(new File(filePath));
	}
	
	/**
	 * 将文件内容读取到预分配好的字节数组中,提高文件读取的速度 <br/>
	 * 注意，linux下可能无法读取到文件大小，不能用此方法读取长度超过最大int值的文件
	 * 
	 * @param file 被读取的文件
	 * @return
	 */
	public static byte[] getFileBytes(File file) throws IOException {
		//需要的字节数为文件的大小
		int fileRestLen = (int)file.length();
		byte[] realData = new byte[fileRestLen];

		InputStream in = null;
		try {
			in = new BufferedInputStream(new FileInputStream(file));
			
			int bufferLen = BUFFER_LENGTH;	//缓冲大小
			if(fileRestLen <= bufferLen) {
				//如果文件大小小于缓冲大小，则一次读取完成
				in.read(realData, 0, fileRestLen);
			} else {
				int len; 		//每次读取到的字节数
				int pos = 0;	//当前读取的位置
				
				//每次读取指定大小字节数，当剩余字节大于此值时，读取剩余的字节
				int readLen = bufferLen;
				while((len = in.read(realData, pos, readLen)) != -1 && readLen > 0) {
					
					pos += len;		//移动读取位置
					fileRestLen -= len;	//文件可读取大小
					if(fileRestLen < bufferLen) {
						readLen = fileRestLen;
					}
				}
			}
		} finally {
			if(null != in) {
				in.close();
			}
		}

		return realData;
	}
	
	public static void main(String[] args) throws Exception {
		File f = new File("G:\\restclient-ui-2.3.3-jar-with-dependencies.jar");
		int time = 100;
		
		//优化后的读取
		long start = System.currentTimeMillis();
		for(int i = 0; i < time; i++)
		FileHelper.getFileBytes(f);
		long total = System.currentTimeMillis() - start;
		System.out.println("nmethod:" + total);
		
		//nio读取
		long start1 = System.currentTimeMillis();
		for(int i = 0; i < time; i++)
		FileOperate.getShareBytes(f);
		long total1 = System.currentTimeMillis() - start1;
		
		System.out.println("nio:" + total1);
	}
}

