/**
 * ILazyMessage.java
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

import java.util.List;

/**
 * ClassName:ILazyMessage <br>
 * Function: 包含元数据的可序列化的消息 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-17  上午9:13:54
 */
public interface ILazyMessage extends ILazySerialize {

	/**
	 * 设置元数据,如果本身已有元数据则此设置将无效
	 * @param
	 * @return
	 */
	void setMetaData(ILazyMetaData metaData);

	/**
	 * 获取元数据
	 * @param
	 * @return
	 */
	ILazyMetaData getMetaData();

	/**
	 * 元数据是由自身产生还是由外部传入的，如果元数据由外部传入，则在序列化时不必序列化元数据(在list的时候使用)
	 * @param
	 * @return 如果元数据由外部传入，则返回true,否则返回false
	 */
	boolean isMetaDataFromOutside();

	void setInt(int columnIndex, int value);

	int getInt(int columnIndex);

	void setLong(int columnIndex, long value);

	long getLong(int columnIndex);

	void setFloat(int columnIndex, float value);

	float getFloat(int columnIndex);

	void setDouble(int columnIndex, double value);

	double getDouble(int columnIndex);

	void setBoolean(int columnIndex, boolean value);

	boolean getBoolean(int columnIndex);

	void setString(int columnIndex, String value);

	String getString(int columnIndex);

	void setBytes(int columnIndex, byte[] value);

	byte[] getBytes(int columnIndex);

	void setMessage(int columnIndex, ILazySerialize value);

	ILazyMessage getMessage(int columnIndex);

	void setList(int columnIndex, List<?> value);

	List<?> getList(int columnIndex);

	void addIntToList(int columnIndex, int value);

	void addLongToList(int columnIndex, long value);

	void addFloatToList(int columnIndex, float value);

	void addDoubleToList(int columnIndex, double value);

	void addBooleanToList(int columnIndex, boolean value);

	void addStringToList(int columnIndex, String value);

	void addBytesToList(int columnIndex, byte[] value);

	void addMessageToList(int columnIndex, ILazySerialize value);

	void addAllToList(int columnIndex, List<?> value);
}

