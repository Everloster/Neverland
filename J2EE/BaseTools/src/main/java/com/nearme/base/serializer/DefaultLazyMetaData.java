/**
 * DefaultLazyMetaData.java
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
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * ClassName:DefaultLazyMetaData <br>
 * Function: 序列化数据的列类型信息 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-15  下午4:49:41
 */
public class DefaultLazyMetaData extends AbstractLazySerialize implements ILazyMetaData {
	private static final long serialVersionUID = 1L;

	private HashMap<Integer, LazySerializeType> columnTypeMap;

	public DefaultLazyMetaData() {
		columnTypeMap = new HashMap<Integer, LazySerializeType>();
	}

	/**
	 * 设置指定列类型，如果该列已设置且与当前设置不同，则设置失败
	 * @param
	 * @return
	 */
	@Override
	public void setColumnType(int columnIndex, LazySerializeType columnType) {
		checkColumnIndex(columnIndex);

		LazySerializeType oldColumnType = columnTypeMap.get(columnIndex);
		if (null == oldColumnType) {
			columnTypeMap.put(columnIndex, columnType);
		} else if (oldColumnType != columnType) {
			//如果指定列已设置且与当前列类型不同，说明定义有问题
			throw new IllegalArgumentException("columnType at " + columnIndex
					+ " already set,old:" + oldColumnType + ",new:" + columnType);
		}
	}

	/**
	 * 获取指定列类型
	 * @param
	 * @return
	 */
	@Override
	public LazySerializeType getColumnType(int columnIndex) {
		LazySerializeType type = columnTypeMap.get(columnIndex);
		if (null == type) {
			type = LazySerializeType.TYPE_UNKNOWN;
		}

		return type;
	}

	@Override
	protected int computeSerializeSize() {
		int size = 0;
		for (Integer key : columnTypeMap.keySet()) {
			LazySerializeType type = columnTypeMap.get(key);
			size += CodedOutputStream.computeInt32Size(key, type.getType());
		}

		return size;
	}

	@Override
	public void writeTo(CodedOutputStream output) throws IOException {
		for (Integer key : columnTypeMap.keySet()) {
			output.writeInt32(key, columnTypeMap.get(key).getType());
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		sb.append("{\n");

		Map<Integer, Object> map = new TreeMap<Integer, Object>(columnTypeMap);
		for (Integer key : map.keySet()) {
			sb.append(key).append(":\t").append(map.get(key)).append('\n');
		}

		sb.append('}');

		return sb.toString();
	}

	@Override
	public DefaultLazyMetaData mergeFrom(byte[] data) throws IOException {
		if (data.length == 0) {
			return this;
		}

		CodedInputStream input = CodedInputStream.newInstance(data);

		return mergeFrom(input);
	}

	@Override
	public DefaultLazyMetaData mergeFrom(InputStream inputStream) throws IOException {
		CodedInputStream input = CodedInputStream.newInstance(inputStream);

		return mergeFrom(input);
	}

	@Override
	public DefaultLazyMetaData mergeFrom(CodedInputStream input) throws IOException {
		while (true) {
			int tag = input.readTag();
			if (tag == 0) {
				return this;
			}

	        int columnIndex = WireFormat.getTagFieldNumber(tag);
	        int type = input.readInt32();
	        setColumnType(columnIndex, LazySerializeTypeHelper.getFromType(type));
		}
	}
}

