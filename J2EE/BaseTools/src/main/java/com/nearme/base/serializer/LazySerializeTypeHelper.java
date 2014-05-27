/**
 * LazySerializeTypeHelper.java
 * com.nearme.base.serializer
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-16 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.serializer;
/**
 * ClassName:LazySerializeTypeHelper <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-16  下午2:49:10
 */
public final class LazySerializeTypeHelper {
	private static final int TYPE_MAX_INDEX = LazySerializeType.values().length;

	/**
	 * 根据序号获取序列化类型
	 * @param
	 * @return
	 */
	public static LazySerializeType getFromType(int type) {
		if (type < 1 || type >= TYPE_MAX_INDEX) {
			return LazySerializeType.TYPE_UNKNOWN;
		}

		return LazySerializeType.values()[type - 1];
	}

	/**
	 * 根据数据类型返回器对应的序列化类型
	 * @param
	 * @return
	 */
	public static LazySerializeType getSerializeType(Object value) {
		if (value instanceof Integer) {
			return LazySerializeType.TYPE_INT;
		} else if (value instanceof Long) {
			return LazySerializeType.TYPE_LONG;
		} else if (value instanceof Float) {
			return LazySerializeType.TYPE_FLOAT;
		} else if (value instanceof Double) {
			return LazySerializeType.TYPE_DOUBLE;
		} else if (value instanceof Boolean) {
			return LazySerializeType.TYPE_BOOLEAN;
		} else if (value instanceof String) {
			return LazySerializeType.TYPE_STRING;
		} else if (value instanceof byte[]) {
			return LazySerializeType.TYPE_BYTES;
		} else if (value instanceof ILazySerialize) {
			return LazySerializeType.TYPE_MESSAGE;
		} else {
			return LazySerializeType.TYPE_UNKNOWN;
		}
	}
}

