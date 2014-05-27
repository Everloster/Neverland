/**
 * MethodUtil.java
 * com.oppo.base.beanutils
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-9 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 */

package com.oppo.base.beanutils;

import java.lang.reflect.Method;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * ClassName:MethodUtil 
 * Function: TODO ADD FUNCTION
 * 
 * @author 80036381
 * @version
 * @since Ver 1.1
 * @Date 2011-8-9 下午04:25:44
 */
public class MethodsUtil {
	private static final Map<String, Method[]> methodMap = new ConcurrentHashMap<String, Method[]>();

	/**
	 * 获取指定类的所有方法
	 * 
	 * @param tClass
	 * @param refresh
	 *            是否重新读取
	 * @return
	 */
	public static Method[] getMethods(Class<?> tClass) {
		return getMethods(tClass, false);
	}

	/**
	 * 获取指定类的所有方法
	 * 
	 * @param tClass
	 * @param refresh
	 *            是否重新读取
	 * @return
	 */
	public static Method[] getMethods(Class<?> tClass, boolean refresh) {
		String className = tClass.getName();
		Method[] methods = null;
		if (!refresh) { // 不用重新读取则先尝试读取已有的
			methods = methodMap.get(className);
		}

		if (refresh || null == methods) {
			methods = tClass.getMethods();
			methodMap.put(className, methods);
		}

		return methods;
	}

	public static Method getMethod(Class<?> tClass, String methodName) {
		Method[] methods = getMethods(tClass);
		for (int i = 0; i < methods.length; i++) {
			Method m = methods[i];
			if (m.getName().equals(methodName)) {
				return m;
			}
		}

		return null;
	}
}
