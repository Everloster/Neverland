/**
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 * FileName:ITransfer.java
 * ProjectName:OStat_Record
 * PackageName:com.oppo.statistic.xml
 * Create Date:2011-5-3
 * History:
 *   ver	date	  author		desc	
 * ────────────────────────────────────────────────────────
 *   1.0	2011-5-3	  80036381		
 *
 * 
*/

package com.oppo.base.beanutils;
/**
 * ClassName:ITransfer
 * Function: 名称转换
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-5-3  上午08:02:49
 */
public interface INameTransfer {
	/**
	 * 从节点名称转换为属性名
	 * @param name 需要转换的名称
	 * @return
	 */
	String transferFromNodeName(String nodeName);
	
	/**
	 * 将属性名称转换为节点名
	 * Function Description here
	 * @param 
	 * @return
	 */
	String transferToNodeName(String name);
	
	/**
	 * 获取指定参数的set方法
	 */
	String getSetMethod(String parameterName);

	/**
	 * 获取指定参数的get方法
	 */
	String getGetMethod(String parameterName);
	
	
}

