/**
 * ITransferValue.java
 * com.oppo.base.beanutils
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-3 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.beanutils;
/**
 * ClassName:ITransferValue
 * Function: 将指定的值装换为指定类的值
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-8-3  下午04:50:35
 */
public interface IValueTransfer<T> {
	Object transferValue(T value, Class<?> destClass);
}

