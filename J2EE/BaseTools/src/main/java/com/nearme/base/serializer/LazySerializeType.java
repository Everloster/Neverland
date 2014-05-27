/**
 * LazySerializeType.java
 * com.nearme.base.serializer
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-15 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.serializer;
/**
 * ClassName:LazySerializeType <br>
 * Function: 序列化类型 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-15  下午5:43:21
 */
public enum LazySerializeType {
	TYPE_INT(1),
	TYPE_LONG(2),
	TYPE_FLOAT(3),
	TYPE_DOUBLE(4),
	TYPE_BOOLEAN(5),
	TYPE_STRING(6),
	TYPE_BYTES(7),
	TYPE_MESSAGE(8),
	TYPE_REPEAT(9),
	TYPE_UNKNOWN(1000);

	private int type;
	LazySerializeType(int type) {
		this.type = type;
	}

	public int getType() {
		return type;
	}
}

