/**
 * ILazyMetaData.java
 * com.nearme.base.serializer
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-17 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.serializer;
/**
 * ClassName:ILazyMetaData <br>
 * Function: 可序列化的元数据 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-17  上午9:15:08
 */
public interface ILazyMetaData extends ILazySerialize {

	/**
	 * 设置指定列类型，如果该列已设置且与当前设置不同，则设置失败
	 * @param
	 * @return
	 */
	void setColumnType(int columnIndex, LazySerializeType columnType);

	/**
	 * 获取指定列类型
	 * @param
	 * @return
	 */
	LazySerializeType getColumnType(int columnIndex);
}

