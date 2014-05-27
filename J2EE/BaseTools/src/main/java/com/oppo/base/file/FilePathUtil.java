/**
 * FilePathUtil.java
 * com.oppo.base.file
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-1 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.file;

import java.io.File;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:FilePathUtil
 * Function: 文件路径相关操作
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-1  上午10:54:35
 */
public class FilePathUtil {
	/**
	 * 将第一次计算得到的值放入的此路径中，后面不再计算
	 */
	private static Map<Class<?>, String> PATH_MAP = new ConcurrentHashMap<Class<?>, String>();
	
	/**
	 * 
	 * 根据指定路径获取与classes同级的路径
	 * @param 
	 * @return
	 */
	public static String getClassPath(Class<?> myClass, String relatePath) {
		String path = PATH_MAP.get(myClass);
		if(null == path) {
			//获取包名
			String packageName = myClass.getPackage().getName().replaceAll("\\.", "/");
			//获取类路径后将包名替换为空
			path = myClass.getResource("").getPath();
			
			//解决中文问题
			try {
				path = URLDecoder.decode(path, OConstants.DEFAULT_ENCODING);
			} catch (UnsupportedEncodingException e) {
			}
			
			path = StringUtil.replaceLast(path, packageName, "");

			PATH_MAP.put(myClass, path);
		}

		//classes上级路径下的对应路径
		return new File(path, relatePath).getAbsolutePath();
	}
}

