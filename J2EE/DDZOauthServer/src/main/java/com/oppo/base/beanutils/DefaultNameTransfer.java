/**
 * DefaultNameTransfer.java
 * com.oppo.base.beanutils
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-4 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.beanutils;
/**
 * ClassName:DefaultNameTransfer
 * Function: 默认的名称转换
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-4  下午02:46:47
 */
public class DefaultNameTransfer implements INameTransfer {

	@Override
	public String transferFromNodeName(String name) {
		return name;
	}
	
	@Override
	public String transferToNodeName(String name) {
		return name.substring(3, 4).toLowerCase() + name.substring(4);
	}

	@Override
	public String getSetMethod(String parameterName) {
		StringBuilder methodBuilder = new StringBuilder(parameterName.length() + 3);
		methodBuilder.append("set");
		methodBuilder.append(parameterName.substring(0, 1).toUpperCase());
		methodBuilder.append(parameterName.substring(1));
		return methodBuilder.toString();
	}

	@Override
	public String getGetMethod(String parameterName) {
		StringBuilder methodBuilder = new StringBuilder(parameterName.length() + 3);
		methodBuilder.append("get");
		methodBuilder.append(parameterName.substring(0, 1).toUpperCase());
		methodBuilder.append(parameterName.substring(1));
		return methodBuilder.toString();
	}
	
	public static void main(String[] args) {
		DefaultNameTransfer dnt = new DefaultNameTransfer();
		System.out.println(dnt.getGetMethod("ab"));
	}
}

