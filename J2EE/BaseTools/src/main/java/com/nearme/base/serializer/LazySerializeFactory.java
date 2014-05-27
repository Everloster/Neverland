/**
 * LazySerializeFactory.java
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

import java.io.IOException;
import java.io.InputStream;

/**
 * ClassName:LazySerializeFactory <br>
 * Function: 序列化数据创建工厂 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-16  下午7:04:12
 */
public final class LazySerializeFactory {

	private LazySerializeFactory() {
	}

	/**
	 * 创建默认的序列化消息
	 * @param
	 * @return
	 */
	public static ILazyMessage createLazyMessage() {
		return new DefaultLazyMessage();
	}

	public static ILazyMessage mergeFrom(byte[] input) throws IOException {
		ILazyMessage msg = createLazyMessage();
		msg.mergeFrom(input);

		return msg;
	}

	public static ILazyMessage mergeFrom(CodedInputStream input) throws IOException {
		ILazyMessage msg = createLazyMessage();
		msg.mergeFrom(input);

		return msg;
	}

	public static ILazyMessage mergeFrom(InputStream input) throws IOException {
		ILazyMessage msg = createLazyMessage();
		msg.mergeFrom(input);

		return msg;
	}

	/**
	 * 创建默认的序列化元数据
	 * @param
	 * @return
	 */
	public static ILazyMetaData createLazyMetaData() {
		return new DefaultLazyMetaData();
	}
}

