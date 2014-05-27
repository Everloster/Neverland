/**
 * ApkStringValueProperty.java
 * com.oppo.base.apk.entity
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-7-25 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.oppo.base.apk.entity;
/**
 * ClassName:ApkStringValueProperty <br>
 * Function: apk中字符资源 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-7-25  上午10:07:05
 */
public class ApkStringValueProperty {

	private int id;
	private String idString;
	private String name;
	private String value;
	/**
	 * 获取id
	 * @return  the id
	 * @since   Ver 1.0
	 */
	public int getId() {
		return id;
	}
	/**
	 * 设置id
	 * @param   id
	 * @since   Ver 1.0
	 */
	public void setId(int id) {
		this.id = id;
	}
	/**
	 * 获取idString
	 * @return  the idString
	 * @since   Ver 1.0
	 */
	public String getIdString() {
		return idString;
	}
	/**
	 * 设置idString
	 * @param   idString
	 * @since   Ver 1.0
	 */
	public void setIdString(String idString) {
		this.idString = idString;
	}
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
		return "id:" + id + ",idString:" + idString
				+ ",name:" + name + ",value:" + value;
	}
}

