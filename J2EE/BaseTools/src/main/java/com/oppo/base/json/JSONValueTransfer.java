/**
 * JSONValueTransfer.java
 * com.oppo.base.json
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-3 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.json;

import java.sql.Date;
import java.sql.Time;
import java.sql.Timestamp;
import java.util.Map;

import com.oppo.base.beanutils.IValueTransfer;
import com.oppo.base.json.JSONObjectParser2;

/**
 * ClassName:JSONValueTransfer
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-8-3  下午04:52:53
 */
public class JSONValueTransfer implements IValueTransfer<String> {
	
	private JSONObjectParser2 jp2 = new JSONObjectParser2();
	
	private Map<String, Class<?>> dClassMap;
	
	public JSONValueTransfer(Map<String, Class<?>> dClassMap) {
		jp2.setStrValueTransfer(this);
		this.dClassMap = dClassMap;
	}

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
		} else if(destClass.equals(String.class)) {
			return value;
		}
		
		try {
			return jp2.getObjectFromJSON(value, destClass, dClassMap);
		} catch(Exception ex) {
			ex.printStackTrace();
			return null;
		}
	}
	
}

