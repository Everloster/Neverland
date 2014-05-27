/**
 * BeanUtil.java
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

import java.lang.reflect.Method;

/**
 * ClassName:BeanUtil Function: 实体操作工具类
 * 
 * @author 80036381
 * @version
 * @since Ver 1.1
 * @Date 2011-7-4 下午02:33:32
 */
public class BeanUtil {
	/**
	 * 设置值到对象
	 * @param nameTransfer	名称转换
	 * @param valueTransfer 值转换
	 * @param obj	需要被设置的对象
	 * @param ms	从指定方法中查找所需的设置方法
	 * @param methodName 未处理的方法名，需要经过nameTransfer转换
	 * @param value	需要设置的值
	 * @return
	 */
	public static <T> boolean setValueToObject(INameTransfer nameTransfer,
			IValueTransfer<T> valueTransfer, Object obj, Method[] ms,
			String methodName, T value) throws Exception {
		//获取设置方法的名称
		String setMethodName = nameTransfer.getSetMethod(methodName);

		Class<?> setParameterClass = null;
		Method setMethod = null;
		for(int i = 0; i < ms.length; i++) {
			Method m = ms[i];
			if(m.getName().equals(setMethodName)) {
				Class<?>[] mClasses = m.getParameterTypes();
				if(mClasses.length == 1) {
					setMethod = m;
					setParameterClass = mClasses[0];
					break;
				}
			}
		}

		if (null != setMethod) {
			setMethod.invoke(obj,
					valueTransfer.transferValue(value, setParameterClass));
			return true;
		} else {
			return false;
		}
	}

	/**
	 * 设置值到对象
	 * @param nameTransfer	名称转换
	 * @param valueTransfer 值转换
	 * @param obj	需要被设置的对象
	 * @param methodName 未处理的方法名，需要经过nameTransfer转换
	 * @param value	需要设置的值
	 * @return
	 */
	public static <T> boolean setValueToObject(INameTransfer nameTransfer,
			IValueTransfer<T> valueTransfer, Object obj, String methodName,
			T value) throws Exception {
		Class<?> tClass = obj.getClass();

		// 通过get方法获取参数类型
		Class<?> setParameterClass = String.class; // 默认为String类型
		String getMethodName = nameTransfer.getGetMethod(methodName);
		Method getMethod = tClass.getMethod(getMethodName, new Class<?>[] {});
		if (null != getMethod) {
			setParameterClass = getMethod.getReturnType();
		}

		// 调用对应方法进行设置
		String setMethodName = nameTransfer.getSetMethod(methodName);
		Method setMethod = tClass.getMethod(setMethodName, setParameterClass);
		if (null != setMethod) {
			setMethod.invoke(obj,
					valueTransfer.transferValue(value, setParameterClass));
			return true;
		} else {
			return false;
		}
	}
}
