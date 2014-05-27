/**
 * ILazySerialize.java
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

import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;

/**
 * ClassName:ILazySerialize <br>
 * Function: 可序列话的数据 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-15  下午6:01:17
 */
public interface ILazySerialize extends Serializable {

	/**
	 * 当前序列化版本
	 * @param
	 * @return
	 */
	int getCurrentVersion();

	/**
	 * 将消息序列化为字节数组<br>
	 * 消息的第一位为总长度，第二位为版本，后面为序列化内容
	 * @param
	 * @return
	 */
	byte[] toByteArray();

	/**
	 * 获取序列化后的大小
	 * @param
	 * @return
	 */
	int getSerializedSize();

	/**
	 * 将消息内容写入到CodedOutputStream中
	 * @param
	 * @return
	 */
	void writeTo(CodedOutputStream output) throws IOException;

	/**
	 * 从CodedInputStream中解析到一个序列化消息
	 * @param
	 * @return
	 */
	ILazySerialize mergeFrom(CodedInputStream input) throws IOException;

	/**
	 * 从字节数组中解析到一个序列化消息
	 * @param
	 * @return
	 */
	ILazySerialize mergeFrom(byte[] input) throws IOException;

	/**
	 * 从InputStream中解析到一个序列化消息
	 * @param
	 * @return
	 */
	ILazySerialize mergeFrom(InputStream input) throws IOException;
}