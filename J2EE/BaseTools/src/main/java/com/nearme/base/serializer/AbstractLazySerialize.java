/**
 * AbstractLazySerialize.java
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
import java.util.List;

/**
 * ClassName:AbstractLazySerialize <br>
 * Function: TODO set FUNCTION <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-15  下午6:03:57
 */
public abstract class AbstractLazySerialize implements ILazySerialize {
	private static final long serialVersionUID = 1L;

	private static final int CURRENT_VERSION = 1;
	protected static final int VERSION_SIZE =  CodedOutputStream.computeInt32SizeNoTag(CURRENT_VERSION);

	private int cacheSerializeSize = -1;

	public AbstractLazySerialize() {
		resetSerializeSize();
	}

	/**
	 * 重置序列化大小，每次参数设置后必须调用
	 * @param
	 * @return
	 */
	protected void resetSerializeSize() {
		cacheSerializeSize = -1;
	}

	/**
	 * 检查列索引是否符合要求（必须大于0）
	 * @param
	 * @return
	 */
	protected void checkColumnIndex(int columnIndex) {
		if (columnIndex <= 0) {
			throw new ArrayIndexOutOfBoundsException("column index must be larger than 0! your column index is " + columnIndex);
		}
	}


	protected int getCacheSerializeSize() {
		return cacheSerializeSize;
	}

	protected void setCacheSerializeSize(int cacheSerializeSize) {
		this.cacheSerializeSize = cacheSerializeSize;
	}

	public int getCurrentVersion() {
		return CURRENT_VERSION;
	}

	@Override
	public int getSerializedSize() {
		//如果缓存size存在则直接返回
		int size = getCacheSerializeSize();
		if (size != -1) {
			return size;
		}

		size = computeSerializeSize();

		this.cacheSerializeSize = size;

		return size;
	}

	/**
	 * 计算序列化的长度
	 * @param
	 * @return
	 */
	protected abstract int computeSerializeSize();

	@Override
	public byte[] toByteArray() {
		try {
			int serializedSize = getSerializedSize();
			if (serializedSize <= 0) {
				return CodedInputStream.EMPTY;
			}

			byte[] flatArray = new byte[serializedSize];
			CodedOutputStream output = CodedOutputStream.newInstance(flatArray);
			writeTo(output);
			output.checkNoSpaceLeft();
			return flatArray;
		} catch (IOException e) {
			throw new RuntimeException(
					"Serializing to a byte throw an IOException (should never happen).",
					e);
		}
	}

	protected int getColumnSize(LazySerializeType lazySerializeType, int columnIndex, Object value) {
		switch(lazySerializeType) {
		case TYPE_INT:
			return CodedOutputStream.computeInt32Size(columnIndex, (Integer)value);
		case TYPE_LONG:
			return CodedOutputStream.computeInt64Size(columnIndex, (Long)value);
		case TYPE_FLOAT:
			return CodedOutputStream.computeFloatSize(columnIndex, (Float)value);
		case TYPE_DOUBLE:
			return CodedOutputStream.computeDoubleSize(columnIndex, (Double)value);
		case TYPE_BOOLEAN:
			return CodedOutputStream.computeBoolSize(columnIndex, (Boolean)value);
		case TYPE_STRING:
			return CodedOutputStream.computeStringSize(columnIndex, (String)value);
		case TYPE_BYTES:
			return CodedOutputStream.computeBytesSize(columnIndex, (byte[])value);
		case TYPE_MESSAGE:
			return CodedOutputStream.computeMessageSize(columnIndex, (ILazySerialize)value);
		case TYPE_REPEAT:
			return CodedOutputStream.computeRepeatSize(columnIndex, (List<?>)value);
		default:
			//无法处理的类型忽略
			return 0;
		}
	}
}

