/**
 * StringValueTransfer.java
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

import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;

/**
 * ClassName:StringValueTransfer
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-8-3  下午04:52:53
 */
public class StringValueTransfer implements IValueTransfer<String> {

	@Override
	public Object transferValue(String value, Class<?> destClass) {
		if(null == value) {
			return null;
		}
		
		if(destClass.equals(int.class)) {			//int
			return Integer.parseInt(value);
		} else if(destClass.equals(long.class)) {	//long
			return Long.parseLong(value);
		} else if(destClass.equals(double.class)) {	//double
			return Double.parseDouble(value);
		} else if(destClass.equals(float.class)) {	//float
			return Float.parseFloat(value);
		} else if(destClass.equals(short.class)) {	//short
			return Short.parseShort(value);
		} else if(destClass.equals(byte.class)) {	//byte
			return Byte.valueOf(value);
		} else if(destClass.equals(Timestamp.class)) {//Timestamp
			return Timestamp.valueOf(value);
		} else if(destClass.equals(Date.class)) {	//Date
			return Date.valueOf(value);
		} else if(destClass.equals(Time.class)) {	//Time
			return Time.valueOf(value);
		} else if(destClass.equals(boolean.class)) {
			return Boolean.parseBoolean(value);
		}
		
		return value;
	}
	
}

