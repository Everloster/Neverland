/**
 * ObjectValueTransfer.java
 * com.oppo.base.beanutils
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-10 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.beanutils;
/**
 * ClassName:ObjectValueTransfer
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-8-10  上午09:18:00
 */
public class ObjectValueTransfer implements IValueTransfer<Object> {

	@Override
	public Object transferValue(Object value, Class<?> destClass) {
		return value;
	}

}

