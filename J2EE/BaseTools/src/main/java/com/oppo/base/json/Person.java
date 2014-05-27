/**
 * Person.java
 * com.oppo.base.json
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-8-10 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.json;
/**
 * ClassName:Person
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-8-10  上午09:34:47
 */
public class Person {
	private String name;
	private String value;
	/**
	 * 获取name
	 * @return  the name
	 * @since   Ver 1.0
	 */
	public String getName() {
		return name;
	}
	/**
	 * 设置name
	 * @param   name    
	 * @since   Ver 1.0
	 */
	public void setName(String name) {
		this.name = name;
	}
	/**
	 * 获取value
	 * @return  the value
	 * @since   Ver 1.0
	 */
	public String getValue() {
		return value;
	}
	/**
	 * 设置value
	 * @param   value    
	 * @since   Ver 1.0
	 */
	public void setValue(String value) {
		this.value = value;
	}
	
	public String toString() {
		return String.format("{name:%s,value:%s}", name, value);
	}
}

