/**
 * IOUtil.java
 * com.oppo.base.common
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-6 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.common;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * ClassName:IOUtil
 * Function: IO相关操作
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-6  上午10:31:00
 */
public class IOUtil {
	/**
	 * 获取流中的字节数组
	 * @param inputStream
	 * @return
	 */
	public static byte[] getBytes(InputStream inputStream) throws IOException {
		return getBytes(inputStream, -1);
	}
	
	/**
	 * 获取流中的指定长度字节数组
	 * @param inputStream
	 * @param dataLen 计划读取的数据长度
	 * @return
	 */
	public static byte[] getBytes(InputStream inputStream, int dataLen) throws IOException {
		int bufferdLen = 4096;
		byte[] data = new byte[bufferdLen];
		
		if(dataLen == -1) {
			dataLen = Integer.MAX_VALUE;
		}
		
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		
		int readSize = Math.min(bufferdLen, dataLen);//计算应该读的大小
		int getSize;	//获取到的字节数
		while(readSize > 0 && (getSize = inputStream.read(data, 0, readSize)) != -1) {
			bos.write(data, 0, getSize);
			dataLen -= getSize;
			readSize = Math.min(bufferdLen, dataLen);
		}
		
		return bos.toByteArray();
	}
}

